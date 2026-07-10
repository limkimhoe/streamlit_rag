import streamlit as st

st.set_page_config(layout="centered", page_title="About", page_icon="ℹ️")

st.title("ℹ️ About This App")

# ── Problem Statement ────────────────────────────────────────────────────────

st.header("🎯 Problem Statement")
st.write("""
Organisations maintain critical knowledge across hundreds of policy documents,
HR handbooks, IT guidelines, and internal web pages. Employees struggle to
locate specific answers, often resorting to time-consuming manual searches or
repeated queries to HR and IT helpdesks.
""")

st.markdown("""
**Key pain points:**
- Policy documents are long and difficult to navigate manually
- Knowledge is scattered across files and web pages with no single entry point
- Employees receive inconsistent answers depending on who they ask
- HR and IT teams spend significant time answering the same repetitive questions
- Employees may act on outdated information when documents are not easy to find
""")

# ── Solution ─────────────────────────────────────────────────────────────────

st.header("💡 Solution")
st.write("""
**Streamlit RAG Chatbot** is a Retrieval-Augmented Generation (RAG) application
that gives employees instant, accurate answers grounded strictly in the
organisation's own knowledge base.
""")

st.markdown("""
An admin uploads policy documents and adds knowledge base URLs through a
protected admin panel. These sources are automatically chunked, embedded into a
vector store, and made searchable. When an employee asks a question, the system:

1. Converts the question into a semantic embedding
2. Retrieves the most relevant passages from the vector store
3. Passes those passages to an LLM as context
4. Returns an answer grounded only in the retrieved content

The chatbot never fabricates answers — if information is not in the knowledge
base, it says so clearly.
""")

# ── Features ─────────────────────────────────────────────────────────────────

st.header("✨ Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**For Users**")
    st.markdown("""
- 💬 Natural language Q&A in a chat interface
- 📚 Source attribution expander for every answer
- 🔄 Streaming responses — see the reply as it generates
- 📊 Real-time token usage tracking
- 🎭 Customisable assistant persona via sidebar
    """)

with col2:
    st.markdown("**For Admins**")
    st.markdown("""
- 📁 Upload PDF, TXT, and MD documents
- 🌐 Add URLs with configurable subpage crawling (BFS)
- ⚡ One-click vector store generation and re-indexing
- 🗑️ Reset and rebuild the knowledge base at any time
- 🔒 Separate admin password independent of user login
    """)

# ── Tech Stack ────────────────────────────────────────────────────────────────

st.header("🛠️ Tech Stack")

st.table({
    "Component": [
        "UI Framework",
        "Language Model",
        "Embeddings",
        "Vector Store",
        "Metadata Store",
        "PDF Extraction",
        "Web Crawling",
        "Token Counting",
    ],
    "Tool": [
        "Streamlit",
        "OpenAI GPT-4o-mini",
        "OpenAI text-embedding-3-small",
        "ChromaDB (persistent, local)",
        "SQLite3",
        "pypdf",
        "requests + BeautifulSoup4",
        "tiktoken",
    ],
})

# ── RAG Pipeline (brief) ─────────────────────────────────────────────────────

st.header("🔄 How RAG Works")

st.image(
    "https://python.langchain.com/assets/images/rag_indexing-8160f90a90a33253d0154659cf7d453f.png",
    caption="RAG: Indexing (left) and Retrieval + Generation (right)",
    use_container_width=True,
) if False else None  # placeholder — remove the `if False` to show an image

st.markdown("""
```
INDEXING (Admin)                    RETRIEVAL (User)
────────────────────                ─────────────────────────────
Upload file / Add URL               User types a question
        │                                   │
        ▼                                   ▼
Extract raw text                    Embed question
        │                           (text-embedding-3-small)
        ▼                                   │
Chunk into ~1500-char               Search ChromaDB
overlapping segments                (cosine similarity)
        │                                   │
        ▼                                   ▼
Embed each chunk                    Top 8 matching chunks
(text-embedding-3-small)                    │
        │                                   ▼
        ▼                           Inject into system prompt
Store in ChromaDB                           │
(with source metadata)                      ▼
                                    GPT-4o-mini generates answer
                                    grounded in retrieved context
```
""")

# ── Team ─────────────────────────────────────────────────────────────────────

st.header("👥 Team")

st.markdown("""
| Name | Role |
|------|------|
| Lim Kim Hoe | Developer — RAG pipeline, vector store, admin panel, chatbot UI |
""")

st.caption(
    "Built as part of an AI Bootcamp project exploring practical RAG applications."
)

# ── Disclaimer ────────────────────────────────────────────────────────────────

st.header("⚠️ Disclaimer")

st.info("""
This application is built for educational and demonstration purposes.
Answers are generated from the indexed knowledge base only and may not reflect
the most current version of any policy. Always verify important decisions
against official policy documents or consult the relevant department directly.
""")
