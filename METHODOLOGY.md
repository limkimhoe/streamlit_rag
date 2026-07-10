# Methodology — Streamlit RAG Chatbot

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Solution Approach](#2-solution-approach)
3. [System Architecture](#3-system-architecture)
4. [Data Ingestion Pipeline](#4-data-ingestion-pipeline)
5. [URL Crawling](#5-url-crawling)
6. [Chunking Strategy](#6-chunking-strategy)
7. [Embedding and Vector Store](#7-embedding-and-vector-store)
8. [Retrieval and Generation](#8-retrieval-and-generation)
9. [Prompt Engineering](#9-prompt-engineering)
10. [Security Design](#10-security-design)
11. [Design Decisions and Trade-offs](#11-design-decisions-and-trade-offs)
12. [Limitations](#12-limitations)

---

## 1. Problem Statement

Organisations maintain critical knowledge across policy documents, HR handbooks,
IT guidelines, and internal web pages. Employees need to consult this content
regularly but face significant friction:

- **Navigation difficulty** — Long PDFs and multi-page websites are slow to search manually.
- **Fragmented knowledge** — Information is spread across multiple file types and URLs with no unified interface.
- **Inconsistent answers** — Verbal answers from colleagues may be outdated or incomplete.
- **Staff overhead** — HR and IT helpdesks spend disproportionate time answering questions already documented in policy.

The target user is an **employee who needs a quick, reliable answer** to a specific question about internal policy, without needing to know which document contains the answer.

---

## 2. Solution Approach

The solution applies **Retrieval-Augmented Generation (RAG)** — a technique that combines semantic search over a private knowledge base with an LLM's language understanding to produce grounded, attributable answers.

RAG was chosen over:

| Alternative | Why RAG is preferred |
|-------------|----------------------|
| Fine-tuning an LLM | Expensive, requires retraining when policies change, prone to hallucination on specifics |
| Pure keyword search | Cannot handle paraphrasing or semantic variation ("annual leave" vs "yearly holidays") |
| LLM with full document in context | Token limits prevent loading entire knowledge bases; cost scales poorly |
| Prompt-stuffing | Unreliable retrieval, no source attribution, context window constraints |

RAG provides:
- **Accuracy** — answers are grounded in retrieved passages, not generated from model memory
- **Attribution** — every answer shows which source excerpt it came from
- **Updatability** — the knowledge base can be refreshed without touching the model
- **Cost efficiency** — only the top-k relevant chunks are sent to the LLM, not entire documents

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        ADMIN PANEL (Page 3)                      │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │ File Upload  │    │ URL Input + max_pages setting        │   │
│  │ (PDF/TXT/MD) │    │ (BFS crawler)                        │   │
│  └──────┬───────┘    └───────────────────┬──────────────────┘   │
│         │                                │                       │
│         ▼                                ▼                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              SQLite3 Metadata Database (files.db)        │   │
│  │   files table: filename, filepath, filesize, indexed     │   │
│  │   urls table:  url, title, max_pages, indexed            │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │ "Generate Vector Store"            │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Indexing Pipeline                      │   │
│  │                                                          │   │
│  │  Extract text → Chunk (1500 chars, 200 overlap)          │   │
│  │       → Embed (text-embedding-3-small)                   │   │
│  │       → Upsert into ChromaDB                             │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                    │
└─────────────────────────────┼────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    ChromaDB     │
                    │  (data/chroma)  │
                    │  Vector Store   │
                    └────────┬────────┘
                             │
┌────────────────────────────┼───────────────────────────────────────┐
│                 CHATBOT (Page 1)                                    │
│                            │                                       │
│  User types question       │                                       │
│        │                   ▼                                       │
│        │    search(query) → embed query                            │
│        │                 → cosine similarity search                │
│        │                 → return top 8 chunks + metadata          │
│        │                            │                              │
│        │                            ▼                              │
│        └──────────────► Build RAG system prompt                   │
│                          (persona + context + instructions)        │
│                                     │                              │
│                                     ▼                              │
│                          GPT-4o-mini (streaming)                  │
│                                     │                              │
│                                     ▼                              │
│                          Display answer + source expander          │
└────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `main.py` | App entry point, user authentication gate |
| `pages/1_💬_Chatbot.py` | Chat UI, RAG orchestration, source attribution |
| `pages/3_🔒_Admin.py` | File/URL management, vector store controls |
| `helper_functions/vectorstore.py` | Full RAG pipeline: ingest → chunk → embed → store → retrieve |
| `helper_functions/llm.py` | OpenAI client wrapper, streaming, token counting |
| `helper_functions/utility.py` | Parameterized password authentication |
| `data/files.db` | SQLite3: tracks sources and indexed status |
| `data/chroma/` | ChromaDB: persisted vector embeddings |

---

## 4. Data Ingestion Pipeline

### File Ingestion

```
Admin uploads file (PDF / TXT / MD)
         │
         ▼
File saved to uploads/ directory
         │
         ▼
add_file() → INSERT into files table (indexed=0)
         │
         ▼ (on "Generate Vector Store")
_extract_file_text()
   ├── PDF  → pypdf.PdfReader → extract page text
   └── TXT/MD → open() with utf-8, errors="ignore"
         │
         ▼
chunk_text(raw_text)
         │
         ▼
_upsert_chunks(collection, filename, chunks, metadata)
         │
         ▼
_mark_file_indexed(filename) → UPDATE files SET indexed=1
```

### Text Extraction Detail

**PDF files** — `pypdf` extracts text page by page. Pages are joined with double newlines to preserve paragraph structure. Note: scanned PDFs (image-based) will produce empty text; OCR is not included.

**TXT / MD files** — read as plain UTF-8. Encoding errors are silently ignored (`errors="ignore"`) to handle files with mixed encodings.

---

## 5. URL Crawling

The URL ingestion uses a **Breadth-First Search (BFS) crawler** that stays within the same domain.

### Algorithm

```
Input: seed_url, max_pages

visited = {}
queue   = [seed_url]

WHILE queue is not empty AND len(visited) < max_pages:
    current = queue.pop(0)          ← BFS: always take from front
    IF current in visited: skip

    visited.add(current)
    response = HTTP GET current
    IF not text/html: skip

    extract and clean text
    collect page text

    FOR each <a href> on page:
        full_url = resolve href relative to current
        IF same scheme + netloc as seed:
            IF not visited and not queued:
                queue.append(full_url)

    sleep(0.3)                      ← polite crawl delay

RETURN seed_title, combined_text
```

### Key Design Choices

| Decision | Reason |
|----------|--------|
| BFS over DFS | BFS explores the site broadly before going deep — better coverage of top-level pages |
| Same-domain restriction | Prevents the crawler from following external links and indexing irrelevant content |
| `max_pages` cap | Prevents runaway crawls on large sites; admin controls the scope |
| Content-Type check | Skips PDFs, images, and downloads — only indexes HTML pages |
| Fragment stripping | `page.html#section` and `page.html` are the same page; stripping `#` prevents duplicate visits |
| 0.3s delay | Polite crawling — avoids overloading servers and reduces risk of being blocked |
| `[Page: url]` prefix | Each sub-page's text is prefixed with its URL so the LLM can attribute statements to specific pages |

### HTML Cleaning

After fetching, these tags are removed before extracting text:
```
<script>  <style>  <nav>  <footer>  <header>  <aside>
```
This removes navigation menus, cookie banners, and footer boilerplate — leaving only meaningful body content.

---

## 6. Chunking Strategy

Raw text (potentially thousands of characters) must be split into segments small enough to embed meaningfully but large enough to contain useful context.

### Parameters

```python
CHUNK_SIZE    = 1500  # characters
CHUNK_OVERLAP = 200   # characters carried into the next chunk
```

### Algorithm — Paragraph-Aware with Overlap

```
Split text on double newlines (\n\n) → list of paragraphs

buffer = ""

FOR each paragraph:

    IF paragraph > CHUNK_SIZE:          ← oversized single paragraph
        emit buffer (if any)
        character-split the paragraph into CHUNK_SIZE slices
        set buffer = last CHUNK_OVERLAP chars of paragraph
        continue

    joined = buffer + "\n\n" + paragraph

    IF len(joined) > CHUNK_SIZE:        ← buffer would overflow
        emit buffer as a chunk
        buffer = buffer[-CHUNK_OVERLAP:] + "\n\n" + paragraph

    ELSE:                               ← fits — keep accumulating
        buffer = joined

emit buffer (final chunk)
```

### Why Paragraph-Aware?

Simple character-splitting cuts mid-sentence, producing fragments that are hard for the LLM to interpret. Paragraph-aware splitting respects natural content boundaries so each chunk is semantically complete.

### Why Overlap?

When a document is split, the boundary between two chunks loses context. With 200-character overlap, consecutive chunks share a small amount of text:

```
Chunk 1: "...employees are entitled to 14 days annual leave per year. This
          applies to all full-time employees regardless of department."

Chunk 2: "department. Employees who have served 5+ years receive 18 days.
          Part-time employees receive leave on a pro-rated basis..."
          ↑ carried from Chunk 1 ──────────────────────────────────
```

This ensures that a question spanning a chunk boundary can still be answered from a single retrieved chunk.

---

## 7. Embedding and Vector Store

### Embedding Model

OpenAI `text-embedding-3-small` converts text into a 1536-dimensional vector that encodes semantic meaning. Semantically similar text produces vectors that are close in this 1536-dimensional space.

```
"annual leave entitlement"    → [0.023, -0.187, 0.441, ...]  (1536 floats)
"how many days off per year"  → [0.019, -0.201, 0.433, ...]  ← close!
"firewall configuration"      → [0.412,  0.093, -0.210, ...]  ← far away
```

### ChromaDB PersistentClient

```python
client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    ),
)
```

ChromaDB handles embedding automatically when documents are added:
- Calls the embedding function on each chunk
- Stores the resulting vectors alongside the text and metadata
- Persists to disk in `data/chroma/` so the store survives restarts

### Chunk Storage Schema

Each chunk is stored with:

| Field | Value | Purpose |
|-------|-------|---------|
| `id` | `"leave_policy.pdf::chunk_3"` | Unique identifier for upsert/delete |
| `document` | Raw chunk text | Returned in search results |
| `metadata.source` | `"leave_policy.pdf"` or URL | Used to group chunks by origin |
| `metadata.type` | `"file"` or `"url"` | Distinguishes source type |
| `metadata.chunk` | `3` | Chunk index within the source |
| `metadata.filename` | `"leave_policy.pdf"` | (file only) original filename |
| `metadata.url` | `"https://..."` | (url only) source URL |
| `metadata.title` | `"MOM Leave"` | (url only) page title |

### Upsert Behaviour

Re-indexing replaces existing chunks rather than creating duplicates:

```python
def _upsert_chunks(collection, source_key, chunks, extra_meta):
    # Delete all existing chunks from this source
    existing = collection.get(where={"source": source_key})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    # Insert fresh chunks
    collection.add(documents=chunks, ids=[...], metadatas=[...])
```

This means clicking "Generate Vector Store" multiple times is safe — it always reflects the latest content.

---

## 8. Retrieval and Generation

### Retrieval — Cosine Similarity Search

When a user submits a question:

```python
results = collection.query(
    query_texts=[user_question],   # ChromaDB embeds this automatically
    n_results=min(8, total_chunks),
    include=["documents", "metadatas"],
)
```

ChromaDB:
1. Embeds the query with the same `text-embedding-3-small` model
2. Computes cosine similarity between the query vector and every stored chunk vector
3. Returns the top 8 chunks with the highest similarity scores

**Cosine similarity** measures the angle between two vectors — a score of 1.0 means identical direction (maximum relevance), 0 means orthogonal (unrelated).

### Generation — LLM with Retrieved Context

The 8 retrieved chunks are joined and injected into the system prompt:

```python
context = "\n\n---\n\n".join(docs)  # join 8 chunks with separators

system_msg = RAG_SYSTEM.format(
    persona=persona,    # admin-configured assistant role
    context=context,    # the 8 retrieved chunks
)

full_messages = [{"role": "system", "content": system_msg}] + conversation_history
response = get_completion_stream(full_messages)
```

The LLM receives the question, the conversation history, and the retrieved chunks — but **not** the entire knowledge base. It synthesises an answer only from what was retrieved.

### Fallback Behaviour

| Scenario | Behaviour |
|----------|-----------|
| Vector store is empty | Return `NOT_IN_KB` immediately, no LLM call |
| Chunks retrieved but unrelated | LLM says it could not find the answer |
| Chunks partially answer the question | LLM shares what it found and notes what is missing |
| Chunks fully answer the question | LLM gives a complete grounded answer |

---

## 9. Prompt Engineering

### RAG System Prompt

```
{persona}

Answer the user's question using the knowledge base excerpts below.
Follow these rules strictly:
1. If the excerpts fully answer the question, give a complete answer.
2. If the excerpts are related but only partially answer the question,
   share what you found and clearly state what information is missing.
3. Only say you could not find an answer if the excerpts are entirely
   unrelated to the question — not just because the answer is incomplete.
Do NOT refuse to engage when the excerpts contain relevant context.

Knowledge base:
{context}
```

### Design Decisions

**Single identity (`{persona}`)** — The assistant's role is set once at the top. Having two "You are" statements (one in persona, one in instructions) causes the model to oscillate between identities and produce inconsistent tone.

**No embedded fallback string** — Earlier versions included the exact "not found" phrase inside the instructions. This primed the model to use that phrase even when context existed. The current prompt instructs the behaviour in natural language without providing the fallback verbatim.

**Three-tier instruction** — A binary "answer or say not found" instruction causes false negatives when chunks are related but not perfectly on-point. The three-tier rule (full answer → partial answer → not found) gives the model a middle option that is more useful to the user.

**`temperature=0`** — Set to zero for maximum determinism. Policy Q&A requires factual consistency; creative variation is undesirable.

### Conversation History

All prior messages are included in `full_messages`, giving the model conversational context:

```python
full_messages = (
    [{"role": "system", "content": system_msg}]
    + st.session_state.messages   # full conversation history
)
```

This allows follow-up questions like "What about for part-time staff?" to be understood in context without repeating the full question.

---

## 10. Security Design

### Dual Password Authentication

Two independent passwords control access:

```python
# Chatbot — default session keys
check_password()

# Admin panel — separate session keys
check_password(
    secret_key="admin_password",
    session_key="admin_correct",
    input_key="admin_input",
)
```

Using separate `session_key` values means logging into the chatbot does not grant admin access, and vice versa. Passwords are stored in `.streamlit/secrets.toml`, which is excluded from version control by `.gitignore`.

### Timing-Safe Comparison

```python
hmac.compare_digest(entered_password, stored_password)
```

Standard `==` comparison short-circuits on the first mismatched character, leaking timing information. `hmac.compare_digest` always takes the same time regardless of where the mismatch occurs, preventing timing-based attacks.

### Secrets Management

| Location | What it contains | In git? |
|----------|-----------------|---------|
| `.streamlit/secrets.toml` | API key, passwords | No (gitignored) |
| `.streamlit/secrets.toml.example` | Empty placeholder with key names | Yes |
| `.env` | Same as secrets.toml for local dev | No (gitignored) |
| `.env.example` | Placeholder | Yes |

---

## 11. Design Decisions and Trade-offs

### Local ChromaDB vs Managed Vector Store

| | ChromaDB (local) | Pinecone / Weaviate (managed) |
|-|-----------------|-------------------------------|
| Cost | Free | Paid (usage-based) |
| Setup | Zero config | API key + cloud account |
| Persistence | Ephemeral on cloud platforms | Persistent |
| Scale | Suitable for < 100K chunks | Millions of vectors |
| **Choice** | ✅ Used (demo/educational project) | Better for production |

ChromaDB was chosen for simplicity and zero cost. On Streamlit Community Cloud, the `data/` directory is reset on reboot — admins must re-index after each deployment. For production use, a managed vector database would be more appropriate.

### pysqlite3-binary Patch

ChromaDB ≥ 0.5 requires SQLite3 ≥ 3.35. Most Linux environments (including the one Streamlit Community Cloud runs on) ship an older SQLite3. The patch replaces the system SQLite3 at import time:

```python
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # modern SQLite3 already available
```

This must execute before `import chromadb`. The `try/except` makes it safe on environments that already have a modern SQLite3.

### Chunk Size Selection

`CHUNK_SIZE=1500` characters balances two competing concerns:

- **Too small** — chunks lose context; a question may need information split across many chunks
- **Too large** — each chunk is less semantically focused; similarity search returns chunks that partially match but contain mostly irrelevant content

1500 characters (~280 words) is approximately one substantial paragraph or policy section — enough context for the LLM to synthesise an answer without noise.

### n_results=8 (Top-k Retrieval)

Retrieving 8 chunks provides sufficient coverage for multi-part questions (e.g., "What are the leave entitlements and how do I apply?") without flooding the LLM context with irrelevant content. At 1500 chars each, 8 chunks is ~12,000 characters — well within GPT-4o-mini's context window.

---

## 12. Limitations

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| Ephemeral storage on Streamlit Cloud | Vector store resets on reboot | Use managed vector DB for production |
| Scanned PDFs (image-based) | No text extracted | Add OCR (e.g., pytesseract) |
| Single knowledge base collection | All sources share one ChromaDB collection | Acceptable for single-tenant use |
| No authentication beyond password | Shared password for all users | Extend with user accounts for multi-user use |
| Crawler may be blocked by some sites | URL returns empty or error | Admin sees error in indexing output |
| No query rewriting | Short queries ("minimum leaves") may retrieve imprecise chunks | Add query expansion or HyDE before retrieval |
| No re-ranking | Top-k by cosine similarity only; may miss nuanced relevance | Add cross-encoder re-ranking step |
