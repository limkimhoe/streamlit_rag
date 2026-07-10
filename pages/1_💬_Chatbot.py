import os
import streamlit as st
from helper_functions.llm import get_completion_stream, count_tokens
from helper_functions.utility import check_password
from helper_functions.vectorstore import (
    collection_count, search, init_db,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(layout="centered", page_title="Chatbot", page_icon="💬")

if not check_password():
    st.stop()

init_db()

# ── API key (mirrors llm.py resolution) ──────────────────────────────────────

if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY", "")
else:
    api_key = st.secrets.get("OPENAI_API_KEY", "")

# ── RAG prompt template ──────────────────────────────────────────────────────
# One clear identity, no embedded fallback string — avoids the priming
# problem where the model repeats the fallback even when context exists.

RAG_SYSTEM = (
    "{persona}\n\n"
    "Answer the user's question using the knowledge base excerpts below.\n"
    "Follow these rules strictly:\n"
    "1. If the excerpts fully answer the question, give a complete answer.\n"
    "2. If the excerpts are related but only partially answer the question, "
    "share what you found and clearly state what information is missing.\n"
    "3. Only say you could not find an answer if the excerpts are entirely "
    "unrelated to the question — not just because the answer is incomplete.\n"
    "Do NOT refuse to engage when the excerpts contain relevant context.\n\n"
    "Knowledge base:\n{context}"
)

NOT_IN_KB = (
    "I'm sorry, I couldn't find a relevant answer "
    "in the knowledge base."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    persona = st.text_area(
        "Assistant Persona",
        value="You are a helpful assistant.",
        height=100,
        help="Describe the assistant's role. This becomes the system prompt.",
    )
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Knowledge Base")
    chunk_count = collection_count(api_key)
    if chunk_count:
        st.success(f"✅ {chunk_count} chunks indexed")
    else:
        st.warning("⚠️ No knowledge base. Ask an admin to index documents.")

    if st.session_state.get("messages"):
        total = sum(
            count_tokens(m["content"])
            for m in st.session_state.messages
        )
        st.metric("Estimated Tokens Used", total)

# ── Chat history ─────────────────────────────────────────────────────────────

st.title("💬 AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input ───────────────────────────────────────────────────────────────

if user_input := st.chat_input("Type your message here..."):
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):

        # 1. Retrieve relevant chunks from the vector store
        try:
            docs, metas = search(user_input, api_key)
        except Exception as exc:
            st.error(f"Vector store error: {exc}")
            st.stop()

        # 2a. No chunks → answer without calling the LLM
        if not docs:
            response = NOT_IN_KB
            st.write(response)

        # 2b. Chunks found → RAG-grounded LLM call
        else:
            context = "\n\n---\n\n".join(docs)
            system_msg = RAG_SYSTEM.format(
                persona=persona, context=context
            )
            full_messages = (
                [{"role": "system", "content": system_msg}]
                + st.session_state.messages
            )
            response = st.write_stream(
                get_completion_stream(full_messages)
            )

        # 3. Source attribution expander
        label = (
            f"📚 {len(docs)} source excerpt(s) used"
            if docs else "📚 No excerpts retrieved"
        )
        with st.expander(label):
            if not docs:
                st.caption(
                    "Nothing matched in the knowledge base for this query."
                )
            for i, (doc, meta) in enumerate(zip(docs, metas), 1):
                source = meta.get("url") or meta.get("filename", "—")
                st.caption(f"Excerpt {i} · {source}")
                st.text(doc[:300])

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
