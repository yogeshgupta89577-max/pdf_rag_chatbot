"""
vector_store.py — PDF ingestion pipeline.

Flow:
    PDF File
      → PyPDFLoader         (load raw text from each page)
      → RecursiveCharacterTextSplitter  (split into overlapping chunks)
      → HuggingFaceEmbeddings           (convert text → vectors, runs locally)
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
)


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Load Gemini Embedding model.
    Uses Google's API (no local model download).
    """

    print("⏳ Loading Gemini Embedding model...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    print("✅ Gemini Embedding model loaded.")

    return embeddings 


# ─────────────────────────────────────────────────────────────────────────────
# PDF Loading & Chunking
# ─────────────────────────────────────────────────────────────────────────────
def load_and_split_pdf(pdf_path: str):
    """
    Load a PDF and split it into overlapping text chunks.

    Why overlap?
        Ensures that context isn't lost at chunk boundaries — a sentence
        that spans two chunks is still fully captured in at least one.
    """
    print(f"📄 Loading PDF: {os.path.basename(pdf_path)}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"   └─ {len(documents)} pages loaded.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split at paragraph → sentence → word → character level
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
    """
    Full ingestion pipeline:
        PDF → chunks → embeddings → FAISS index (saved to disk)
    """
    chunks    = load_and_split_pdf(pdf_path)
    embeddings = get_embeddings()

    print("⚡ Building FAISS vector store…")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"✅ Vector store saved → {VECTOR_STORE_PATH}")
    return vectorstore


def load_vector_store() -> FAISS:
    """Load an existing FAISS index from disk (avoids re-embedding)."""
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"No vector store found at '{VECTOR_STORE_PATH}'. "
            "Call create_vector_store() first."
        )
    embeddings  = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,   # required by FAISS for local files
    )
    print(f"✅ Vector store loaded from {VECTOR_STORE_PATH}")
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────
def get_retriever(vectorstore: FAISS):
    """
    Wrap the vector store in a LangChain Retriever.
    search_type='similarity'  →  cosine-similarity search
    k=TOP_K_RESULTS           →  return the 4 most relevant chunks
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )
