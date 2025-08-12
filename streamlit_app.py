# streamlit_app.py

# DO NOT REMOVE: Fix for ChromaDB on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import traceback
from operator import itemgetter
from typing import List

# --- Tenacity for Retrying ---
from tenacity import retry, stop_after_attempt, wait_fixed

# --- LangChain & AI Imports ---
from huggingface_hub import login
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain.schema.output_parser import StrOutputParser

# --- Document Processing Imports ---
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import fitz
from docx import Document

# --- Page Configuration ---
st.set_page_config(page_title="Project Al-Kharizmi", layout="wide")

# --- Configuration & Secrets ---
HF_TOKEN = st.secrets.get("HF_TOKEN")
project_root = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')
CHROMA_PATH = os.path.join(project_root, 'chroma')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# --- THE ROBUST SOLUTION: A RETRYING EMBEDDER ---
# This class inherits from the original and adds smarter retry logic.
class RetryingHuggingFaceInferenceAPIEmbeddings(HuggingFaceInferenceAPIEmbeddings):
    @retry(
        wait=wait_fixed(8),
        stop=stop_after_attempt(12)
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Call the original, error-prone function
        result = super().embed_documents(texts)
        # **THE FINAL FIX**: Check for an empty response and treat it as an error to trigger a retry.
        if not result:
            raise ValueError("Hugging Face API returned an empty list of embeddings. Retrying...")
        return result


# --- Caching RAG Components (Lazy Loading) ---
@st.cache_resource
def initialize_rag_components():
    st.write("Initializing RAG components for the first time... This may take a moment.")
    try:
        if not HF_TOKEN:
            st.error("HF_TOKEN secret not found. Please set it in the Streamlit app settings.")
            return None, None, None
        
        login(token=HF_TOKEN)

        # Use our new, robust embedder
        embeddings = RetryingHuggingFaceInferenceAPIEmbeddings(
            api_key=HF_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
        )
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})

        PROMPT_TEMPLATE = """
        Answer the question based only on the following context:
        {context}
        ---
        Answer the question based on the above context: {question}
        """
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-large", huggingfacehub_api_token=HF_TOKEN
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = RunnableParallel({
            "answer": (
                {"context": lambda x: format_docs(x["context"]), "question": itemgetter("question")}
                | prompt
                | llm
                | StrOutputParser()
            ),
            "context": itemgetter("context"),
        })
        st.success("RAG components initialized successfully!")
        return db, rag_chain, retriever
    except Exception as e:
        st.error(f"Could not initialize RAG components: {e}")
        traceback.print_exc()
        return None, None, None

# --- Helper functions ---
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'jpg', 'jpeg'}
def allowed_file(filename): return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(filepath):
    extension = filepath.rsplit('.', 1)[1].lower()
    text = ""
    try:
        if extension == 'pdf':
            with fitz.open(filepath) as doc:
                for page in doc: text += page.get_text()
        elif extension == 'docx':
            doc = Document(filepath)
            for para in doc.paragraphs: text += para.text + '\n'
        elif extension in ['jpg', 'jpeg']:
            text = pytesseract.image_to_string(Image.open(filepath))
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None


# --- Main Application Logic ---
db, rag_chain, retriever = initialize_rag_components()
tab1, tab2 = st.tabs(["AI Assistant (Al-Kharizmi)", "Knowledge Base Manager (Scribe)"])

# --- Tab 1: AI Assistant ---
with tab1:
    st.header("Project Al-Kharizmi")
    st.write("Ask a question about the documents in the knowledge base.")
    question = st.text_area("Your Question:", key="question_area")
    if st.button("Ask The Assistant", key="ask_button"):
        if not question:
            st.warning("Please enter a question.")
        elif rag_chain is None or retriever is None or db is None:
            st.error("RAG components are not available. Check initialization.")
        else:
            try:
                existing_docs = db.get(include=[]) 
                if not existing_docs or not existing_docs['ids']:
                    st.warning("The knowledge base is empty. Please upload a document in the 'Knowledge Base Manager' tab first.")
                else:
                    with st.spinner("Step 1 of 2: Searching knowledge base..."):
                        context_docs = retriever.invoke(question)
                    with st.spinner("Step 2 of 2: Generating answer with AI..."):
                        response_data = rag_chain.invoke({"context": context_docs, "question": question})
                        st.subheader("Answer:")
                        st.write(response_data['answer'])
                        st.subheader("Sources Used:")
                        for doc in response_data['context']:
                            with st.expander(f"Source: {doc.metadata.get('source', 'Unknown')}"):
                                st.write(doc.page_content)
            except Exception as e:
                st.error(f"An error occurred while running the query: {e}")
                traceback.print_exc()


# --- Tab 2: Knowledge Base Manager ---
with tab2:
    st.header("Project Scribe")
    st.write("Manage the documents in the AI's knowledge base.")
    uploaded_file = st.file_uploader("Upload a new document (PDF, DOCX, JPG)", type=["pdf", "docx", "jpg", "jpeg"])
    if uploaded_file is not None:
        filename = secure_filename(uploaded_file.name)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "wb") as f: f.write(uploaded_file.getbuffer())
        with st.spinner(f"Processing '{filename}'... This may take a minute if the model is waking up."):
            raw_text = extract_text_from_file(filepath)
            if raw_text is None:
                st.error("Could not extract text from the file.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                documents = text_splitter.create_documents(texts=[raw_text], metadatas=[{"source": filename}])
                try:
                    if db is not None:
                        # This call is now protected by the smarter retry logic
                        db.add_documents(documents=documents)
                        st.success(f"Success! Processed '{filename}' and added {len(documents)} chunks to the knowledge base.")
                        st.rerun()
                except Exception as e:
                    # This will now catch the error cleanly if all retries fail
                    st.error(f"Could not add document to database. The AI model might be temporarily unavailable. The error was: {e}")
                    traceback.print_exc()
    st.subheader("Current Knowledge Base")
    if db is not None:
        try:
            retrieved_docs = db.get(include=["metadatas"])
            if not retrieved_docs or not retrieved_docs['ids']:
                st.info("The knowledge base is empty. Upload a document to get started.")
            else:
                source_counts = {}
                for metadata in retrieved_docs['metadatas']:
                    source = metadata.get('source', 'Unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1
                for source, count in source_counts.items():
                    col1, col2 = st.columns([4, 1])
                    with col1: st.write(f"**{source}** ({count} chunks)")
                    with col2:
                        if st.button("Delete", key=f"delete_{source}"):
                            try:
                                db.delete(where={"source": source})
                                source_filepath = os.path.join(UPLOAD_FOLDER, source)
                                if os.path.exists(source_filepath): os.remove(source_filepath)
                                st.success(f"Deleted '{source}' from the knowledge base.")
                                st.rerun()
                            except Exception as e: st.error(f"Failed to delete '{source}': {e}")
        except Exception as e:
            st.error(f"Could not retrieve documents from the database. It might be empty or corrupted. Error: {e}")