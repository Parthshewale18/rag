# 📚 RAG — Retrieval-Augmented Generation Pipeline

A conversational **Retrieval-Augmented Generation (RAG)** pipeline built with [LangChain](https://www.langchain.com/), [ChromaDB](https://www.trychroma.com/), and [Google Gemini](https://ai.google.dev/). Ingest text documents, store them as vector embeddings, and ask natural-language questions — with full conversation-history awareness.

---

## ✨ Features

- **Document Ingestion** — Recursively load `.txt` files, chunk them with configurable size/overlap, and persist embeddings in a local ChromaDB vector store.
- **Semantic Retrieval** — Retrieve the most relevant document chunks for any query using similarity or MMR search.
- **LLM-Powered Answers** — Pass retrieved context to Google Gemini (`gemini-2.5-flash`) to generate concise, grounded answers.
- **History-Aware Conversations** — Multi-turn chatbot that reformulates follow-up questions into standalone queries so retrieval stays accurate across turns.

---

## 🏗️ Architecture

```
docs/*.txt
    │
    ▼
┌──────────────┐   chunks    ┌──────────────────┐
│  ingestion.py │ ─────────▶ │  ChromaDB (local) │
└──────────────┘             │  + HuggingFace    │
                             │    Embeddings     │
                             └────────┬─────────┘
                                      │ retrieval
                                      ▼
                             ┌──────────────────┐
                             │  retrieval.py /   │
                             │  history_aware.py │
                             │  (LangChain +     │
                             │   Gemini LLM)     │
                             └──────────────────┘
                                      │
                                      ▼
                               Generated Answer
```

---

## 📂 Project Structure

```
rag/
├── docs/                  # Source documents to ingest (e.g. openai.txt)
├── chroma_db/             # Persisted ChromaDB vector store (auto-generated)
├── ingestion.py           # Load, chunk, and embed documents into ChromaDB
├── retrieval.py           # One-shot query: retrieve docs → generate answer
├── history_aware.py       # Multi-turn chatbot with history-aware retrieval
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not committed)
├── .gitignore
└── README.md
```

| File | Purpose |
|---|---|
| [`ingestion.py`](ingestion.py) | Loads `.txt` files from `docs/`, splits them into chunks (default 1 000 chars, 200 overlap), embeds with `all-MiniLM-L6-v2`, and stores in ChromaDB. |
| [`retrieval.py`](retrieval.py) | Runs a single query against the vector store, retrieves the top-3 similar chunks, and generates an answer via Gemini. |
| [`history_aware.py`](history_aware.py) | Interactive chatbot loop. Reformulates each question using conversation history before retrieval, then answers using retrieved context. |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- A **Google AI API key** (for Gemini) — [get one here](https://aistudio.google.com/apikey)

### 1. Clone the Repository

```bash
git clone https://github.com/Parthshewale18/rag.git
cd rag
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Add Documents

Place your `.txt` files inside the `docs/` directory. A sample file (`openai.txt`) is included to get you started.

---

## 💡 Usage

### Step 1 — Ingest Documents

Run the ingestion script to load, chunk, and embed your documents:

```bash
python ingestion.py
```

You should see output like:

```
Loading files from docs...
Loaded 1 documents.
Chunking documents into chunks of size 1000 with overlap 200...
Created 15 chunks.
Creating vector store 'my_collection'...
Vector store created successfully.
Ingestion complete.
```

### Step 2a — Single Query (Retrieval)

Run a one-off question against the vector store:

```bash
python retrieval.py
```

> **Note:** Edit the `query` variable in `retrieval.py` to change the question.

### Step 2b — Interactive Chat (History-Aware)

Start a multi-turn conversational session:

```bash
python history_aware.py
```

```
History-Aware RAG Chatbot
Type 'history' to view the conversation, 'quit' to exit.

You : What research areas does OpenAI work in?
Bot : OpenAI works in ...

You : Tell me more about the second one.
Bot : ...
```

The chatbot automatically reformulates follow-up questions (like *"Tell me more about the second one"*) into standalone queries so retrieval remains accurate.

---

## 🔧 Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `chunk_size` | `ingestion.py` | `1000` | Max characters per chunk |
| `chunk_overlap` | `ingestion.py` | `200` | Overlapping characters between chunks |
| `collection_name` | `ingestion.py` | `my_collection` | ChromaDB collection name |
| `search_type` | `retrieval.py` / `history_aware.py` | `similarity` / `mmr` | Retrieval strategy |
| `k` | `retrieval.py` / `history_aware.py` | `3` | Number of documents to retrieve |
| `temperature` | `retrieval.py` / `history_aware.py` | `0.7` | LLM creativity (0 = deterministic) |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Framework | [LangChain](https://www.langchain.com/) |
| Embeddings | [HuggingFace `all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| LLM | [Google Gemini 2.5 Flash](https://ai.google.dev/) |
| Environment | [python-dotenv](https://pypi.org/project/python-dotenv/) |

---

## 📄 License

This project is open-source. Feel free to use and modify it for your own purposes.
