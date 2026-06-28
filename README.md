# 🧠 PDF Intelligence Chatbot
### Built with LangChain + RAG + FAISS + Streamlit

Upload any PDF → Get an auto-generated structured summary → Chat with it using RAG.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📤 PDF Upload | Drag & drop any PDF file |
| 📋 Auto Summary | map-reduce summarization via LangChain |
| 💬 RAG Q&A | Retrieves relevant chunks, then answers with LLM |
| 🧠 Structured Output | Every answer has Direct Answer / Explanation / Source / Takeaway |
| 🔁 Memory | Follow-up questions remember previous context |
| 🆓 Free Embeddings | HuggingFace all-MiniLM-L6-v2 (runs locally) |
| 🆓 Free LLM | Groq (llama3-8b-8192) — no cost |

---

## 🏗️ Project Structure

```
pdf_rag_chatbot/
├── app.py                ← Streamlit UI (run this)
├── rag_pipeline.py       ← ConversationalRetrievalChain + Memory
├── summarizer.py         ← load_summarize_chain (map_reduce)
├── vector_store.py       ← PyPDFLoader → TextSplitter → FAISS
├── prompt_templates.py   ← All LangChain PromptTemplates
├── config.py             ← Constants, model names, chunk sizes
├── requirements.txt
├── .env.example          ← Copy to .env and add your keys
└── README.md
```

---

## 🚀 Setup & Run

### 1. Clone / download the project
```bash
cd pdf_rag_chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
cp .env.example .env
# Open .env and add your Groq key
# Get a FREE key at: https://console.groq.com
```

### 5. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser. 🎉

---

## 🔄 How It Works (RAG Pipeline)

```
PDF Upload
   │
   ▼
PyPDFLoader → load all pages as Documents
   │
   ▼
RecursiveCharacterTextSplitter → chunk_size=1000, overlap=200
   │
   ▼
HuggingFaceEmbeddings → 384-dim vectors (local, free)
   │
   ▼
FAISS Vector Store → stored on disk at ./faiss_index
   │
   ══════════  Query Time  ══════════
   │
   ❓ User Question
   │
   ▼
Similarity Search → top-4 most relevant chunks
   │
   ▼
PromptTemplate → [context] + [chat_history] + [question]
   │
   ▼
LLM (Groq / OpenAI) → generates structured answer
   │
   ▼
📊 Output:
   📌 Direct Answer
   📖 Explanation
   🔍 Source Reference
   💡 Key Takeaway
```

---

## 🧩 LangChain Components Used

| Component | Module | Purpose |
|---|---|---|
| `PyPDFLoader` | `langchain_community.document_loaders` | Load PDF pages |
| `RecursiveCharacterTextSplitter` | `langchain_text_splitters` | Chunk documents |
| `HuggingFaceEmbeddings` | `langchain_huggingface` | Create vectors |
| `FAISS` | `langchain_community.vectorstores` | Store & search vectors |
| `ConversationalRetrievalChain` | `langchain.chains` | Full RAG chain |
| `ConversationBufferMemory` | `langchain.memory` | Store chat history |
| `load_summarize_chain` | `langchain.chains.summarize` | Map-reduce summary |
| `PromptTemplate` | `langchain_core.prompts` | Custom prompts |
| `ChatGroq` | `langchain_groq` | Groq LLM |

---

## 🔧 Customisation

**Change chunk size** → edit `config.py`
```python
CHUNK_SIZE    = 1000   # increase for more context per chunk
CHUNK_OVERLAP = 200    # increase for better boundary handling
```

**Change number of retrieved chunks** → edit `config.py`
```python
TOP_K_RESULTS = 4      # increase for broader context (may slow down)
```

**Change the answer format** → edit `prompt_templates.py`
Modify `QA_PROMPT_TEMPLATE` to add / remove sections.

**Switch LLM** → use the sidebar dropdown in the app or change:
```python
GROQ_MODEL   = "llama-3.1-8b-instant"   # bigger, smarter Groq model
```

---

## 📝 Notes

- First run downloads the embedding model (~90 MB) — this is cached after that.
- Large PDFs (100+ pages) may take 1–2 minutes to process.
- The FAISS index is saved to `./faiss_index/` — delete it to force re-indexing.
- Conversation memory is in-session only (cleared on page refresh).

---

## 🎓 Learning Outcomes

After building this project you will understand:
- How RAG works end-to-end
- Why we chunk documents and what overlap does
- How embeddings represent meaning as vectors
- How FAISS performs similarity search
- How LangChain chains compose multiple components
- How conversation memory works in multi-turn chatbots
- How map-reduce summarization handles long documents

---

Built as a learning project while studying **LangChain + RAG** 🚀
