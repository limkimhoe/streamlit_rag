import os
import streamlit as st
from pathlib import Path
from helper_functions.utility import check_password
from helper_functions.vectorstore import (
    UPLOADS_DIR,
    add_file, delete_file, get_all_files,
    add_url, delete_url, get_all_urls,
    index_all_sources, reset_vector_store,
    collection_count, init_db,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(layout="centered", page_title="Admin", page_icon="🔒")

if not check_password(
    secret_key="admin_password",
    session_key="admin_correct",
    input_key="admin_input",
):
    st.stop()

init_db()

# ── API key ──────────────────────────────────────────────────────────────────

if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY", "")
else:
    api_key = st.secrets.get("OPENAI_API_KEY", "")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.success("✅ Logged in as Admin")
    if st.button("🚪 Logout"):
        st.session_state["admin_correct"] = False
        st.rerun()

# ── Title ────────────────────────────────────────────────────────────────────

st.title("🔒 Admin Panel")

# ════════════════════════════════════════════════════════════════════════════
# Section 1 — File Upload
# ════════════════════════════════════════════════════════════════════════════

st.header("📁 Document Upload")

uploaded = st.file_uploader(
    "Upload PDF, TXT, or MD files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)
if uploaded:
    for f in uploaded:
        dest = UPLOADS_DIR / f.name
        dest.write_bytes(f.read())
        add_file(f.name, str(dest), f.size)
    st.success(f"Uploaded {len(uploaded)} file(s).")
    st.rerun()

files = get_all_files()
if not files:
    st.info("No files uploaded yet.")
else:
    for row in files:
        c1, c2, c3, c4 = st.columns([4, 1, 2, 1])
        c1.write(row["filename"])
        c2.write(f"{(row['file_size'] or 0) / 1024:.1f} KB")
        c3.write("✅ Indexed" if row["indexed"] else "⏳ Pending")
        if c4.button("🗑️", key=f"del_f_{row['filename']}", help="Delete"):
            p = Path(row["filepath"])
            if p.exists():
                p.unlink()
            delete_file(row["filename"])
            st.rerun()

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Section 2 — URL Knowledge Base
# ════════════════════════════════════════════════════════════════════════════

st.header("🌐 URL Knowledge Base")
st.caption("Add web pages as an alternative knowledge source.")

with st.form("add_url_form", clear_on_submit=True):
    new_url = st.text_input(
        "Page URL",
        placeholder="https://example.com/policy",
    )
    max_pages = st.number_input(
        "Max pages to crawl",
        min_value=1, max_value=50, value=1, step=1,
        help=(
            "1 = only the exact URL. "
            "Higher values follow same-domain links via BFS. "
            "Large sites: keep this low (5-20) to avoid long indexing times."
        ),
    )
    if st.form_submit_button("➕ Add URL") and new_url:
        if not new_url.startswith(("http://", "https://")):
            st.error("URL must start with http:// or https://")
        else:
            add_url(new_url.strip(), max_pages=int(max_pages))
            pages_label = f"{max_pages} page(s)"
            st.success(f"Added: {new_url} (crawl up to {pages_label})")
            st.rerun()

urls = get_all_urls()
if not urls:
    st.info("No URLs added yet.")
else:
    for row in urls:
        c1, c2, c3, c4 = st.columns([4, 1, 2, 1])
        label = row["title"] or row["url"]
        c1.write(label[:55] + "…" if len(label) > 55 else label)
        c2.caption(f"≤{row['max_pages']} pg")
        c3.write("✅ Indexed" if row["indexed"] else "⏳ Pending")
        if c4.button("🗑️", key=f"del_u_{row['url']}", help="Remove"):
            delete_url(row["url"])
            st.rerun()

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Section 3 — Vector Store
# ════════════════════════════════════════════════════════════════════════════

st.header("🗄️ Vector Store")

chunk_total = collection_count(api_key)
total_sources = len(files) + len(urls)
indexed_sources = (
    sum(1 for r in files if r["indexed"]) +
    sum(1 for r in urls if r["indexed"])
)

m1, m2, m3 = st.columns(3)
m1.metric("Total Sources", total_sources)
m2.metric("Indexed", indexed_sources)
m3.metric("Chunks in Store", chunk_total)

st.caption(
    "**Generate** embeds all files and URLs into ChromaDB. "
    "Re-running re-indexes everything so edits are always reflected."
)

col_gen, col_reset = st.columns(2)

with col_gen:
    if st.button(
        "⚡ Generate Vector Store",
        type="primary",
        disabled=not total_sources,
        use_container_width=True,
    ):
        with st.spinner("Chunking and embedding — this may take a moment…"):
            count, errors = index_all_sources(api_key)
        if count:
            st.success(f"✅ Indexed {count} source(s).")
        for err in errors:
            st.error(err)
        st.rerun()

with col_reset:
    if st.button(
        "🗑️ Reset Vector Store",
        type="secondary",
        disabled=not chunk_total,
        use_container_width=True,
    ):
        reset_vector_store(api_key)
        st.warning("Vector store cleared. Click Generate to rebuild.")
        st.rerun()
