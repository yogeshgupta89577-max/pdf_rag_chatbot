"""
config.py — Central configuration for PDF RAG Chatbot
All settings, model names, and constants live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Provider: Groq ──────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"   # Fixed model, hardcoded

# ── Embeddings (Google Gemini) ──────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ── Text Splitter ─────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000   # characters per chunk
CHUNK_OVERLAP = 200    # overlap between consecutive chunks

# ── Retriever ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS = 4      # number of similar chunks to retrieve per query

# ── Vector Store ──────────────────────────────────────────────────────────────
VECTOR_STORE_PATH = "./faiss_index"