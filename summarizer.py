"""
summarizer.py — Auto-generates a structured PDF summary using map-reduce.

Strategy  (map_reduce):
    MAP    → each chunk gets summarized individually by the LLM
    REDUCE → all chunk-summaries are combined into one final summary

Why map_reduce instead of 'stuff'?
    'stuff' puts ALL text into a single LLM call — it hits token limits fast.
    map_reduce handles documents of any length.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq

from config import GROQ_MODEL, GROQ_API_KEY
from prompt_templates import SUMMARY_MAP_PROMPT, SUMMARY_COMBINE_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory  (Groq only — inbuilt, no user input needed)
# ─────────────────────────────────────────────────────────────────────────────
def get_llm():
    """
    Return ChatGroq LLM using the hardcoded model and .env API key.

    Model  : llama-3.3-70b-versatile (set in config.py)
    API Key: GROQ_API_KEY from .env file
    """
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.3,
        max_tokens=2048,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Summary Function
# ─────────────────────────────────────────────────────────────────────────────
def generate_summary(pdf_path: str) -> str:
    """
    Load a PDF and return a structured Markdown-formatted summary.

    Steps:
        1. Load PDF pages
        2. Split into larger chunks (3 000 chars — more context per map call)
        3. Sample up to 8 chunks evenly across the document
        4. Run map_reduce summarization chain
        5. Return final structured summary string
    """
    print("\n📋 Generating summary…")

    # ── 1. Load PDF ─────────────────────────────────────────────────────────
    loader    = PyPDFLoader(pdf_path)
    documents = loader.load()

    # ── 2. Split into larger chunks for summarization ────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # ── 3. Sample evenly (avoid token-limit issues on large PDFs) ───────────
    max_chunks = 8
    if len(chunks) > max_chunks:
        step   = len(chunks) // max_chunks
        chunks = chunks[::step][:max_chunks]

    print(f"   └─ Summarising {len(chunks)} document sections…")

    # ── 4. Build map-reduce chain ────────────────────────────────────────────
    llm = get_llm()

    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=SUMMARY_MAP_PROMPT,
        combine_prompt=SUMMARY_COMBINE_PROMPT,
        verbose=False,
    )

    # ── 5. Run & return ──────────────────────────────────────────────────────
    result  = chain.invoke({"input_documents": chunks})
    summary = result["output_text"]

    print("✅ Summary generated!")
    return summary
