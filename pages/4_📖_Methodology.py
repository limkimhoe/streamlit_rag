import streamlit as st

st.set_page_config(layout="centered", page_title="Methodology", page_icon="📖")

st.title("📖 Methodology")
st.caption("Technical documentation for the Streamlit RAG Chatbot system.")

# ── Table of Contents ────────────────────────────────────────────────────────

with st.expander("📋 Table of Contents", expanded=False):
    st.markdown("""
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
    """)

st.divider()

# ── 1. Problem Statement ─────────────────────────────────────────────────────

st.header("1. Problem Statement")

st.write("""
Organisations maintain critical knowledge across policy documents, HR handbooks,
IT guidelines, and internal web pages. Employees need to consult this content
regularly but face significant friction:
""")

col1, col2 = st.columns(2)
with col1:
    st.error("**Navigation difficulty**\nLong PDFs and multi-page websites are slow to search manually.")
    st.error("**Fragmented knowledge**\nInformation is spread across multiple file types and URLs with no unified interface.")
with col2:
    st.error("**Inconsistent answers**\nVerbal answers from colleagues may be outdated or incomplete.")
    st.error("**Staff overhead**\nHR and IT helpdesks spend disproportionate time on questions already documented in policy.")

st.info(
    "**Target user:** an employee who needs a quick, reliable answer to a specific "
    "policy question — without knowing which document contains the answer."
)

st.divider()

# ── 2. Solution Approach ─────────────────────────────────────────────────────

st.header("2. Solution Approach")

st.write("""
The solution applies **Retrieval-Augmented Generation (RAG)** — a technique that
combines semantic search over a private knowledge base with an LLM's language
understanding to produce grounded, attributable answers.
""")

st.subheader("Why RAG over alternatives?")

st.table({
    "Alternative": [
        "Fine-tuning an LLM",
        "Pure keyword search",
        "LLM with full document in context",
        "Prompt-stuffing",
    ],
    "Why RAG is preferred": [
        "Expensive; requires retraining when policies change; prone to hallucination on specifics",
        "Cannot handle paraphrasing or semantic variation (\"annual leave\" vs \"yearly holidays\")",
        "Token limits prevent loading entire knowledge bases; cost scales poorly",
        "Unreliable retrieval, no source attribution, context window constraints",
    ],
})

st.subheader("RAG provides:")

c1, c2, c3, c4 = st.columns(4)
c1.success("**Accuracy**\nAnswers grounded in retrieved passages, not model memory")
c2.success("**Attribution**\nEvery answer shows which source excerpt it came from")
c3.success("**Updatability**\nKnowledge base refreshed without touching the model")
c4.success("**Cost efficiency**\nOnly top-k relevant chunks sent to the LLM")

st.divider()

# ── 3. System Architecture ───────────────────────────────────────────────────

st.header("3. System Architecture")

st.code("""
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
│  │  Extract text → Chunk (1500 chars, 200 overlap)          │   │
│  │       → Embed (text-embedding-3-small)                   │   │
│  │       → Upsert into ChromaDB                             │   │
│  └──────────────────────────┬───────────────────────────────┘   │
└─────────────────────────────┼────────────────────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │    ChromaDB     │
                    │  (data/chroma)  │
                    │  Vector Store   │
                    └────────┬────────┘
                             │
┌────────────────────────────┼───────────────────────────────────────┐
│                 CHATBOT (Page 1)                                    │
│  User types question       │                                       │
│        │                   ▼                                       │
│        │    search(query) → embed query                            │
│        │                 → cosine similarity search                │
│        │                 → return top 8 chunks + metadata          │
│        │                            │                              │
│        └──────────────► Build RAG system prompt                   │
│                          (persona + context + instructions)        │
│                                     ▼                              │
│                          GPT-4o-mini (streaming)                  │
│                                     ▼                              │
│                          Display answer + source expander          │
└────────────────────────────────────────────────────────────────────┘
""", language=None)

st.subheader("Component Responsibilities")

st.table({
    "Component": [
        "main.py",
        "pages/1_💬_Chatbot.py",
        "pages/3_🔒_Admin.py",
        "helper_functions/vectorstore.py",
        "helper_functions/llm.py",
        "helper_functions/utility.py",
        "data/files.db",
        "data/chroma/",
    ],
    "Responsibility": [
        "App entry point, user authentication gate",
        "Chat UI, RAG orchestration, source attribution",
        "File/URL management, vector store controls",
        "Full RAG pipeline: ingest → chunk → embed → store → retrieve",
        "OpenAI client wrapper, streaming, token counting",
        "Parameterized password authentication",
        "SQLite3: tracks sources and indexed status",
        "ChromaDB: persisted vector embeddings",
    ],
})

st.divider()

# ── 4. Data Ingestion Pipeline ───────────────────────────────────────────────

st.header("4. Data Ingestion Pipeline")

st.subheader("File Ingestion Flow")

st.code("""
Admin uploads file (PDF / TXT / MD)
         │
         ▼
File saved to uploads/ directory
         │
         ▼
add_file() → INSERT into files table (indexed=0)
         │
         ▼  (on "Generate Vector Store")
_extract_file_text()
   ├── PDF   → pypdf.PdfReader → extract page text
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
""", language=None)

st.subheader("Text Extraction Detail")

st.markdown("""
- **PDF files** — `pypdf` extracts text page by page. Pages are joined with
  double newlines to preserve paragraph structure. Scanned (image-based) PDFs
  produce empty text — OCR is not included.
- **TXT / MD files** — read as plain UTF-8. Encoding errors are silently
  ignored (`errors="ignore"`) to handle files with mixed encodings.
""")

st.divider()

# ── 5. URL Crawling ──────────────────────────────────────────────────────────

st.header("5. URL Crawling")

st.write("""
URL ingestion uses a **Breadth-First Search (BFS) crawler** that stays
within the same domain as the seed URL.
""")

st.subheader("BFS Algorithm")

st.code("""
Input: seed_url, max_pages

visited = {}
queue   = [seed_url]

WHILE queue is not empty AND len(visited) < max_pages:
    current = queue.pop(0)          ← BFS: always take from the front
    IF current in visited: skip

    visited.add(current)
    response = HTTP GET current
    IF not text/html: skip

    extract and clean page text
    append to collected texts

    FOR each <a href> on the page:
        full_url = resolve href relative to current
        IF same scheme + netloc as seed:
            IF not visited and not queued:
                queue.append(full_url)

    sleep(0.3)                      ← polite inter-request delay

RETURN seed_title, combined_text
""", language=None)

st.subheader("Key Design Choices")

st.table({
    "Decision": [
        "BFS over DFS",
        "Same-domain restriction",
        "`max_pages` cap",
        "Content-Type check",
        "Fragment stripping",
        "0.3 s delay",
        "`[Page: url]` prefix",
    ],
    "Reason": [
        "BFS explores broadly before going deep — better coverage of top-level pages",
        "Prevents the crawler following external links and indexing irrelevant content",
        "Prevents runaway crawls on large sites; admin controls the scope",
        "Skips PDFs, images, and downloads — only indexes HTML pages",
        "`page.html#section` and `page.html` are the same page; stripping `#` avoids duplicate visits",
        "Polite crawling — avoids overloading servers and reduces risk of being blocked",
        "Each sub-page's text is prefixed with its URL so the LLM can attribute statements to specific pages",
    ],
})

st.subheader("HTML Cleaning")
st.write("These tags are removed before extracting text, stripping navigation and boilerplate:")
st.code("<script>  <style>  <nav>  <footer>  <header>  <aside>", language="html")

st.divider()

# ── 6. Chunking Strategy ─────────────────────────────────────────────────────

st.header("6. Chunking Strategy")

st.write("""
Raw text (potentially thousands of characters) must be split into segments
small enough to embed meaningfully but large enough to contain useful context.
""")

st.subheader("Parameters")
c1, c2 = st.columns(2)
c1.metric("Chunk Size", "1 500 chars", help="~280 words — one substantial paragraph or policy section")
c2.metric("Chunk Overlap", "200 chars", help="Characters carried from the end of one chunk into the next")

st.subheader("Paragraph-Aware Algorithm with Overlap")

st.code("""
Split text on double newlines (\\n\\n) → list of paragraphs

buffer = ""

FOR each paragraph:

    IF paragraph > CHUNK_SIZE:          ← oversized single paragraph
        emit buffer (if any)
        character-split the paragraph into CHUNK_SIZE slices
        set buffer = last CHUNK_OVERLAP chars of paragraph
        continue

    joined = buffer + "\\n\\n" + paragraph

    IF len(joined) > CHUNK_SIZE:        ← buffer would overflow
        emit buffer as a chunk
        buffer = buffer[-CHUNK_OVERLAP:] + "\\n\\n" + paragraph

    ELSE:                               ← fits — keep accumulating
        buffer = joined

emit buffer (final chunk)
""", language=None)

st.subheader("Why paragraph-aware?")
st.write("""
Simple character-splitting cuts mid-sentence, producing fragments that are
hard for the LLM to interpret. Paragraph-aware splitting respects natural
content boundaries so each chunk is semantically complete.
""")

st.subheader("Why overlap?")
st.write("With 200-character overlap, consecutive chunks share a small amount of text:")

st.code("""
Chunk 1: "...employees are entitled to 14 days annual leave per year. This
          applies to all full-time employees regardless of department."

Chunk 2: "department. Employees who have served 5+ years receive 18 days.
          Part-time employees receive leave on a pro-rated basis..."
          ↑ carried from Chunk 1 ──────────────────────────────────
""", language=None)

st.write("""
This ensures that a question spanning a chunk boundary can still be answered
from a single retrieved chunk.
""")

st.divider()

# ── 7. Embedding and Vector Store ────────────────────────────────────────────

st.header("7. Embedding and Vector Store")

st.subheader("Embedding Model — text-embedding-3-small")

st.write("""
OpenAI `text-embedding-3-small` converts text into a **1536-dimensional
vector** that encodes semantic meaning. Semantically similar text produces
vectors that are close in this high-dimensional space.
""")

st.code("""
"annual leave entitlement"    → [0.023, -0.187, 0.441, ...]  (1536 floats)
"how many days off per year"  → [0.019, -0.201, 0.433, ...]  ← semantically close
"firewall configuration"      → [0.412,  0.093, -0.210, ...]  ← semantically distant
""", language=None)

st.subheader("ChromaDB PersistentClient Setup")

st.code("""
client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    ),
)
""", language="python")

st.subheader("Chunk Storage Schema")

st.table({
    "Field": [
        "id", "document", "metadata.source",
        "metadata.type", "metadata.chunk",
        "metadata.filename", "metadata.url", "metadata.title",
    ],
    "Example value": [
        '"leave_policy.pdf::chunk_3"', "Raw chunk text",
        '"leave_policy.pdf" or URL',
        '"file" or "url"', "3",
        '"leave_policy.pdf"', '"https://..."', '"MOM Leave"',
    ],
    "Purpose": [
        "Unique ID for upsert/delete", "Returned in search results",
        "Groups chunks by origin",
        "Distinguishes source type", "Chunk index within the source",
        "(file only) Original filename",
        "(url only) Source URL", "(url only) Page title",
    ],
})

st.subheader("Upsert Behaviour")

st.write("""
Re-indexing replaces existing chunks rather than creating duplicates.
Clicking **Generate Vector Store** multiple times is always safe.
""")

st.code("""
def _upsert_chunks(collection, source_key, chunks, extra_meta):
    # Delete all existing chunks from this source first
    existing = collection.get(where={"source": source_key})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    # Insert fresh chunks
    collection.add(documents=chunks, ids=[...], metadatas=[...])
""", language="python")

st.divider()

# ── 8. Retrieval and Generation ──────────────────────────────────────────────

st.header("8. Retrieval and Generation")

st.subheader("Retrieval — Cosine Similarity Search")

st.code("""
results = collection.query(
    query_texts=[user_question],   # ChromaDB embeds this automatically
    n_results=min(8, total_chunks),
    include=["documents", "metadatas"],
)
""", language="python")

st.write("""
ChromaDB:
1. Embeds the query using the same `text-embedding-3-small` model
2. Computes **cosine similarity** between the query vector and every stored chunk vector
3. Returns the top 8 chunks with the highest similarity scores

Cosine similarity measures the angle between two vectors — a score of **1.0**
means identical direction (maximum relevance); **0** means orthogonal (unrelated).
""")

st.subheader("Generation — LLM with Retrieved Context")

st.code("""
context = "\\n\\n---\\n\\n".join(docs)   # join top 8 chunks with separators

system_msg = RAG_SYSTEM.format(
    persona=persona,    # admin-configured assistant role
    context=context,    # the 8 retrieved chunks
)

full_messages = [{"role": "system", "content": system_msg}] + conversation_history
response = get_completion_stream(full_messages)
""", language="python")

st.write("""
The LLM receives the question, conversation history, and retrieved chunks —
but **not** the entire knowledge base. It synthesises an answer only from
what was retrieved.
""")

st.subheader("Fallback Behaviour")

st.table({
    "Scenario": [
        "Vector store is empty",
        "Chunks retrieved but unrelated",
        "Chunks partially answer the question",
        "Chunks fully answer the question",
    ],
    "Behaviour": [
        "Return NOT_IN_KB immediately — no LLM call made",
        "LLM says it could not find the answer",
        "LLM shares what it found and notes what is missing",
        "LLM gives a complete, grounded answer",
    ],
})

st.divider()

# ── 9. Prompt Engineering ────────────────────────────────────────────────────

st.header("9. Prompt Engineering")

st.subheader("RAG System Prompt")

st.code("""
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
""", language=None)

st.subheader("Design Decisions")

with st.expander("Single identity (`{persona}`)"):
    st.write("""
    The assistant's role is set once at the top. Having two "You are"
    statements — one in the persona, one in the instructions — causes the model
    to oscillate between identities and produce inconsistent tone.
    """)

with st.expander("No embedded fallback string"):
    st.write("""
    Earlier versions included the exact "not found" phrase inside the
    instructions. This primed the model to use that phrase even when context
    existed. The current prompt instructs the behaviour in natural language
    without providing the fallback verbatim.
    """)

with st.expander("Three-tier instruction (full → partial → not found)"):
    st.write("""
    A binary "answer or say not found" instruction causes false negatives when
    chunks are related but not perfectly on-point. The three-tier rule gives
    the model a middle option that is more useful to the user.
    """)

with st.expander("`temperature=0`"):
    st.write("""
    Set to zero for maximum determinism. Policy Q&A requires factual
    consistency; creative variation is undesirable.
    """)

st.subheader("Conversation History")

st.write("""
All prior messages are included in `full_messages`, giving the model
conversational context. This allows follow-up questions like
*"What about for part-time staff?"* to be understood without repeating
the full question.
""")

st.code("""
full_messages = (
    [{"role": "system", "content": system_msg}]
    + st.session_state.messages   # full conversation history
)
""", language="python")

st.divider()

# ── 10. Security Design ──────────────────────────────────────────────────────

st.header("10. Security Design")

st.subheader("Dual Password Authentication")

st.write("""
Two independent passwords control access, using separate session state keys
so that logging into the chatbot does not grant admin access, and vice versa.
""")

st.code("""
# Chatbot page — uses default session keys
check_password()

# Admin panel — uses entirely separate session keys
check_password(
    secret_key="admin_password",   # reads st.secrets["admin_password"]
    session_key="admin_correct",   # independent st.session_state key
    input_key="admin_input",       # independent widget key
)
""", language="python")

st.subheader("Timing-Safe Comparison")

st.write("""
Standard `==` comparison short-circuits on the first mismatched character,
leaking timing information that can be exploited. `hmac.compare_digest`
always takes the same time regardless of where the mismatch occurs.
""")

st.code("hmac.compare_digest(entered_password, stored_password)", language="python")

st.subheader("Secrets Management")

st.table({
    "Location": [
        ".streamlit/secrets.toml",
        ".streamlit/secrets.toml.example",
        ".env",
        ".env.example",
    ],
    "What it contains": [
        "API key, passwords",
        "Empty placeholder with key names",
        "Same as secrets.toml for local dev",
        "Placeholder",
    ],
    "Committed to git?": ["No (gitignored)", "Yes", "No (gitignored)", "Yes"],
})

st.divider()

# ── 11. Design Decisions and Trade-offs ──────────────────────────────────────

st.header("11. Design Decisions and Trade-offs")

st.subheader("Local ChromaDB vs Managed Vector Store")

st.table({
    "": ["Cost", "Setup", "Persistence", "Scale", "Choice"],
    "ChromaDB (local)": [
        "Free",
        "Zero config",
        "Ephemeral on cloud platforms",
        "< 100 K chunks",
        "✅ Used (demo/educational project)",
    ],
    "Pinecone / Weaviate (managed)": [
        "Paid (usage-based)",
        "API key + cloud account",
        "Persistent",
        "Millions of vectors",
        "Better for production",
    ],
})

st.write("""
ChromaDB was chosen for simplicity and zero cost. On Streamlit Community
Cloud the `data/` directory resets on reboot, so admins must re-index
after each deployment. A managed vector database is recommended for production.
""")

st.subheader("pysqlite3-binary Patch")

st.write("""
ChromaDB ≥ 0.5 requires SQLite3 ≥ 3.35. Most Linux environments ship an
older version. The patch replaces the system SQLite3 at import time, before
`import chromadb` runs:
""")

st.code("""
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # modern SQLite3 already available — no action needed
""", language="python")

st.subheader("Chunk Size — 1 500 characters")

st.write("""
- **Too small** — chunks lose context; a single question may need information
  split across many chunks, none of which are individually sufficient.
- **Too large** — each chunk is less semantically focused; search returns chunks
  that partially match but contain mostly irrelevant content.

1 500 characters (~280 words) approximates one substantial paragraph or policy
section — enough context for the LLM to synthesise an answer without noise.
""")

st.subheader("Top-k = 8 chunks")

st.write("""
Retrieving 8 chunks provides coverage for multi-part questions without
flooding the LLM context with irrelevant content. At 1 500 chars each,
8 chunks is ~12 000 characters — well within GPT-4o-mini's context window.
""")

st.divider()

# ── 12. Limitations ──────────────────────────────────────────────────────────

st.header("12. Limitations")

st.table({
    "Limitation": [
        "Ephemeral storage on Streamlit Cloud",
        "Scanned PDFs (image-based)",
        "Single knowledge base collection",
        "No user-level authentication",
        "Crawler may be blocked by some sites",
        "No query rewriting",
        "No re-ranking",
    ],
    "Impact": [
        "Vector store resets on reboot",
        "No text extracted",
        "All sources share one ChromaDB collection",
        "Shared password for all users of a role",
        "URL returns empty or error during indexing",
        "Short queries may retrieve imprecise chunks",
        "Top-k by cosine similarity only; may miss nuanced relevance",
    ],
    "Mitigation": [
        "Use a managed vector DB (Pinecone, Weaviate) for production",
        "Add OCR (e.g., pytesseract) for scanned document support",
        "Acceptable for single-tenant; add namespacing for multi-tenant",
        "Extend with individual user accounts for production use",
        "Admin sees error in indexing output; try a different URL",
        "Add query expansion or HyDE (Hypothetical Document Embeddings)",
        "Add cross-encoder re-ranking step after initial retrieval",
    ],
})
