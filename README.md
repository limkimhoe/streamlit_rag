# Streamlit RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, OpenAI, and ChromaDB. The chatbot answers questions strictly from a knowledge base built by an admin — combining uploaded documents and crawled web pages.

---

## What It Does

| Role | Capability |
|------|-----------|
| **User** | Ask questions; get answers grounded in the knowledge base |
| **Admin** | Upload documents, add URLs, generate and manage the vector store |

If the knowledge base does not contain relevant information, the chatbot says so rather than hallucinating an answer.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                       ADMIN PANEL                       │
│  Upload PDFs/TXT  ──►  SQLite3 (files table)           │
│  Add URLs         ──►  SQLite3 (urls table)            │
│                                                         │
│  Generate Vector Store                                  │
│    ├── Files: extract text ──► chunk ──► embed          │
│    └── URLs:  BFS crawl   ──► chunk ──► embed          │
│                             └──────────────────────►   │
│                                    ChromaDB             │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│                       CHATBOT                           │
│  User question ──► embed query                         │
│                ──► similarity search (top 8 chunks)    │
│                ──► inject chunks into system prompt    │
│                ──► GPT-4o-mini answers from context    │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
week-08_improved/
├── main.py                          # App entry point + user login
├── requirements.txt
├── .env.example                     # Local development env template
├── .gitignore
│
├── .streamlit/
│   ├── secrets.toml.example         # Secrets template (copy → secrets.toml)
│   └── secrets.toml                 # ← gitignored, never committed
│
├── pages/
│   ├── 1_💬_Chatbot.py             # RAG chatbot UI
│   ├── 2_ℹ️_About.py               # About page
│   └── 3_🔒_Admin.py               # Admin panel (password protected)
│
├── helper_functions/
│   ├── llm.py                       # OpenAI client, streaming, token counting
│   ├── utility.py                   # Password check helper
│   └── vectorstore.py               # Core RAG pipeline (SQLite3 + ChromaDB)
│
├── sample_docs/                     # Ready-to-upload demo knowledge base
│   ├── leave_policy.pdf
│   ├── it_security_policy.pdf
│   └── remote_work_policy.pdf
│
├── data/                            # ← gitignored (auto-created at runtime)
│   ├── files.db                     # SQLite3 metadata database
│   └── chroma/                      # ChromaDB vector store
│
└── uploads/                         # ← gitignored (uploaded files stored here)
```

---

## Key Components

### 1. `helper_functions/vectorstore.py`

The core of the RAG system. Handles everything from storage to retrieval.

#### SQLite3 — Metadata Tracking

Two tables track what has been added and whether it has been indexed:

```
files table                          urls table
────────────────────────────         ────────────────────────────
id          INTEGER PRIMARY KEY      id          INTEGER PRIMARY KEY
filename    TEXT UNIQUE              url         TEXT UNIQUE
filepath    TEXT                     title       TEXT
file_size   INTEGER                  max_pages   INTEGER
uploaded_at TEXT                     added_at    TEXT
indexed     INTEGER (0/1)            indexed     INTEGER (0/1)
indexed_at  TEXT                     indexed_at  TEXT
```

#### ChromaDB — Vector Store

Chunks are stored as embeddings using OpenAI's `text-embedding-3-small` model:

```python
embedding_fn = OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small",
)
client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection("documents", embedding_function=embedding_fn)
```

Each chunk is stored with metadata so search results can be attributed to their source:

```python
# Example metadata for a URL chunk
{"source": "https://www.mom.gov.sg/leave", "type": "url", "title": "Leave - MOM", "chunk": 3}

# Example metadata for a file chunk
{"source": "leave_policy.pdf", "type": "file", "filename": "leave_policy.pdf", "chunk": 0}
```

---

### 2. URL Crawling — BFS Algorithm

When an admin adds a URL with `max_pages > 1`, the crawler follows links using Breadth-First Search (BFS):

```
Seed URL: https://www.mom.gov.sg/employment-practices/leave  (max_pages=5)

Queue: [seed]
Step 1: Visit seed → extract text → discover 10 same-domain links → queue them
Queue: [link1, link2, link3, link4, ...]

Step 2: Visit link1 → extract text → discover more links → queue new ones
...

Stop when visited == max_pages (5 pages total)
```

**Safety rules built into the crawler:**
- Only follows links on the **same domain** (never leaves `mom.gov.sg`)
- Skips non-HTML responses (PDFs, images, downloads)
- Strips navigation, headers, footers, and scripts — only keeps body text
- Adds a 0.3-second delay between requests (polite crawling)
- Each visited page's text is prefixed with `[Page: <url>]` for attribution

---

### 3. Text Chunking — Paragraph-Aware

Raw text is split into overlapping chunks that respect paragraph boundaries:

```
CHUNK_SIZE    = 1500 characters
CHUNK_OVERLAP = 200 characters

Paragraph 1 (400 chars) ─┐
Paragraph 2 (600 chars)  ├─► Chunk 1 (1000 chars)
Paragraph 3 (300 chars) ─┘

         ↑200 chars overlap↓

Paragraph 3 (300 chars) ─┐
Paragraph 4 (800 chars)  ├─► Chunk 2 (1100 chars)
Paragraph 5 (200 chars) ─┘
```

The 200-character overlap means consecutive chunks share context, so meaning is never lost at a boundary.

---

### 4. RAG Prompt Design

The system prompt follows a three-tier instruction:

```
1. If the excerpts fully answer the question → give a complete answer
2. If the excerpts are related but partial   → share what was found, note what's missing
3. If the excerpts are entirely unrelated    → say "not found"
```

This prevents the model from refusing to answer when it has *partial* but relevant information.

---

### 5. `helper_functions/llm.py`

Wraps the OpenAI client with two call modes:

| Function | Use |
|----------|-----|
| `get_completion(messages)` | Single blocking call, returns full string |
| `get_completion_stream(messages)` | Streaming generator for `st.write_stream()` — shows tokens as they arrive |
| `count_tokens(text)` | Estimates token count via `tiktoken` for the sidebar usage metric |

---

### 6. `helper_functions/utility.py`

`check_password()` is parameterized to support **two independent password sessions**:

```python
# Chatbot page — uses default keys
check_password()

# Admin page — uses separate keys so admin login doesn't unlock user session
check_password(
    secret_key="admin_password",   # reads from secrets.toml
    session_key="admin_correct",   # independent session state key
    input_key="admin_input",       # independent widget key
)
```

Passwords are compared with `hmac.compare_digest()` to prevent timing attacks.

---

## Setup

### Local Development

1. **Clone the repo**
   ```bash
   git clone git@github.com:limkimhoe/streamlit_rag.git
   cd streamlit_rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure secrets**
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   # Edit secrets.toml and fill in your values
   ```

   `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "sk-..."
   password       = "your-user-password"
   admin_password = "your-admin-password"
   ```

4. **Run the app**
   ```bash
   streamlit run main.py
   ```

---

### Streamlit Community Cloud

1. Push code to GitHub (secrets are never committed)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Main file path** to `main.py`
4. Under **Advanced settings → Secrets**, paste your `secrets.toml` content
5. Click **Deploy**

To update secrets later: app dashboard → **⋮ → Settings → Secrets → Save** (app reboots automatically).

> **Note:** The `data/` and `uploads/` directories are ephemeral on Community Cloud. The admin must re-upload documents and regenerate the vector store after each reboot.

---

## Usage

### As a User

1. Log in with the user password
2. Navigate to **💬 Chatbot**
3. Optionally customise the assistant persona in the sidebar
4. Type a question — the chatbot answers from the knowledge base only

### As an Admin

1. Navigate to **🔒 Admin** and log in with the admin password
2. **Upload documents** — PDF, TXT, or MD files
3. **Add URLs** — paste a page URL and set how many subpages to crawl
4. Click **⚡ Generate Vector Store** to embed everything into ChromaDB
5. The chatbot is immediately ready to answer from the new content

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `openai` | GPT-4o-mini completions + `text-embedding-3-small` embeddings |
| `chromadb>=0.5.0` | Local persistent vector store |
| `pysqlite3-binary` | Modern SQLite3 required by ChromaDB (patches system SQLite3) |
| `pypdf>=4.0.0` | PDF text extraction |
| `requests` | HTTP fetching for URL crawling |
| `beautifulsoup4` | HTML parsing and text extraction |
| `tiktoken` | Token counting for usage metrics |
| `python-dotenv` | Local `.env` file support |
| `pandas` | Data utilities |
