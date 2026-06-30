"""
vector_store.py — PDF ingestion pipeline.

Flow:
    PDF File
      → PyPDFLoader         (load raw text from each page)
      → RecursiveCharacterTextSplitter  (split into overlapping chunks)
      → GoogleGenerativeAIEmbeddings    (convert text → vectors via Gemini API)
      → FAISS Vector Store              (store & search vectors efficiently)
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
    VECTOR_STORE_PATH,
    GOOGLE_API_KEY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Load Gemini Embedding model using Google's API.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("❌ GOOGLE_API_KEY missing in .env file! Required for Gemini Embeddings.")

    print("⏳ Loading Gemini Embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    print("✅ Gemini Embedding model loaded.")
    return embeddings 


# ─────────────────────────────────────────────────────────────────────────────
# PDF Loading & Chunking
# ─────────────────────────────────────────────────────────────────────────────
def load_and_split_pdf(pdf_path: str):
    """Load a PDF and split it into overlapping text chunks."""
    print(f"📄 Loading PDF: {os.path.basename(pdf_path)}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"   └─ {len(documents)} pages loaded.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"   └─ {len(chunks)} chunks created "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store — Create / Save / Load
# ─────────────────────────────────────────────────────────────────────────────
def create_vector_store(pdf_path: str) -> FAISS:
    """Full Ingestion Pipeline: PDF → chunks → embeddings → FAISS index."""
    chunks    = load_and_split_pdf(pdf_path)
    embeddings = get_embeddings()

    print("⚡ Building FAISS vector store…")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"✅ Vector store saved → {VECTOR_STORE_PATH}")
    return vectorstore


def load_vector_store() -> FAISS:
    """Load an existing FAISS index from disk."""
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"No vector store found at '{VECTOR_STORE_PATH}'. "
            "Call create_vector_store() first."
        )
    embeddings  = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"✅ Vector store loaded from {VECTOR_STORE_PATH}")
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────
def get_retriever(vectorstore: FAISS):
    """Wrap the vector store in a LangChain Retriever."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )