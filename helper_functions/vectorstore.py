"""Vector store helpers.

SQLite3  → tracks file + URL metadata (name/url, indexed status).
ChromaDB → stores text chunk embeddings (also backed by SQLite3 internally).
"""
from __future__ import annotations

# pysqlite3-binary bundles a modern SQLite3 (≥3.35) required by ChromaDB.
# The swap must happen before `import chromadb`.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
UPLOADS_DIR = _ROOT / "uploads"
DB_PATH = DATA_DIR / "files.db"
CHROMA_PATH = str(DATA_DIR / "chroma")
COLLECTION_NAME = "documents"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)


# ── SQLite3 ──────────────────────────────────────────────────────────────────

@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create files and urls tables if they don't exist."""
    _ensure_dirs()
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL UNIQUE,
                filepath    TEXT NOT NULL,
                file_size   INTEGER,
                uploaded_at TEXT DEFAULT (datetime('now')),
                indexed     INTEGER DEFAULT 0,
                indexed_at  TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS urls (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                url        TEXT NOT NULL UNIQUE,
                title      TEXT DEFAULT '',
                added_at   TEXT DEFAULT (datetime('now')),
                indexed    INTEGER DEFAULT 0,
                indexed_at TEXT,
                max_pages  INTEGER DEFAULT 1
            )
        """)
        # Migration: add max_pages to databases created before this column existed
        try:
            conn.execute("ALTER TABLE urls ADD COLUMN max_pages INTEGER DEFAULT 1")
        except Exception:
            pass


# ── File helpers ─────────────────────────────────────────────────────────────

def add_file(filename: str, filepath: str, file_size: int) -> None:
    with _db() as conn:
        conn.execute(
            """INSERT INTO files (filename, filepath, file_size, indexed)
               VALUES (?, ?, ?, 0)
               ON CONFLICT(filename) DO UPDATE SET
                   filepath=excluded.filepath,
                   file_size=excluded.file_size,
                   indexed=0, indexed_at=NULL""",
            (filename, filepath, file_size),
        )


def delete_file(filename: str) -> None:
    with _db() as conn:
        conn.execute("DELETE FROM files WHERE filename = ?", (filename,))


def get_all_files() -> list:
    with _db() as conn:
        return conn.execute(
            "SELECT * FROM files ORDER BY uploaded_at DESC"
        ).fetchall()


def _mark_file_indexed(filename: str) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE files SET indexed=1, indexed_at=datetime('now')"
            " WHERE filename=?",
            (filename,),
        )


# ── URL helpers ──────────────────────────────────────────────────────────────

def add_url(url: str, max_pages: int = 1) -> None:
    with _db() as conn:
        conn.execute(
            """INSERT INTO urls (url, max_pages) VALUES (?, ?)
               ON CONFLICT(url) DO UPDATE SET max_pages=excluded.max_pages""",
            (url, max_pages),
        )


def delete_url(url: str) -> None:
    with _db() as conn:
        conn.execute("DELETE FROM urls WHERE url = ?", (url,))


def get_all_urls() -> list:
    with _db() as conn:
        return conn.execute(
            "SELECT * FROM urls ORDER BY added_at DESC"
        ).fetchall()


def _mark_url_indexed(url: str, title: str) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE urls SET indexed=1, indexed_at=datetime('now'), title=?"
            " WHERE url=?",
            (title, url),
        )


# ── Text extraction ──────────────────────────────────────────────────────────

def _extract_file_text(filepath: str) -> str:
    path = Path(filepath)
    if path.suffix.lower() == ".pdf":
        import pypdf
        with open(filepath, "rb") as f:
            reader = pypdf.PdfReader(f)
            return "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def crawl_url(url: str, max_pages: int = 1) -> Tuple[str, str]:
    """BFS-crawl *url* and up to *max_pages* same-domain pages.

    Returns (seed_title, combined_text).  Each sub-page's text is prefixed
    with its URL so the LLM can attribute claims to specific pages.
    """
    import time
    from urllib.parse import urljoin, urlparse

    import requests
    from bs4 import BeautifulSoup

    base = urlparse(url)
    base_origin = f"{base.scheme}://{base.netloc}"
    headers = {"User-Agent": "Mozilla/5.0"}

    visited: set = set()
    queue: list[str] = [url]
    all_texts: list[str] = []
    seed_title = url

    while queue and len(visited) < max_pages:
        current = urlparse(queue.pop(0))._replace(fragment="").geturl()
        if current in visited:
            continue
        visited.add(current)

        try:
            resp = requests.get(current, timeout=15, headers=headers)
            resp.raise_for_status()
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
        except Exception:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        if len(visited) == 1:  # seed page — capture title
            seed_title = (soup.title.string or url).strip() if soup.title else url

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines()]
        text = "\n\n".join(ln for ln in lines if ln)
        if text.strip():
            all_texts.append(f"[Page: {current}]\n\n{text}")

        # Discover same-domain links for remaining quota
        if len(visited) < max_pages:
            for a in soup.find_all("a", href=True):
                full = urlparse(urljoin(current, a["href"]))
                clean = full._replace(fragment="").geturl()
                if (
                    full.scheme in ("http", "https")
                    and f"{full.scheme}://{full.netloc}" == base_origin
                    and clean not in visited
                    and clean not in queue
                ):
                    queue.append(clean)
            if queue:
                time.sleep(0.3)  # polite inter-request delay

    return seed_title, "\n\n\n".join(all_texts)


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> List[str]:
    """Paragraph-aware overlapping chunker.

    Splits on blank lines first so chunks respect natural paragraph
    boundaries. Falls back to character-splitting for paragraphs that
    are individually longer than CHUNK_SIZE.
    """
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buffer = ""

    for para in paragraphs:
        # A single paragraph longer than CHUNK_SIZE: character-split it
        if len(para) > CHUNK_SIZE:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            for i in range(0, len(para), CHUNK_SIZE - CHUNK_OVERLAP):
                chunks.append(para[i: i + CHUNK_SIZE])
            buffer = para[-CHUNK_OVERLAP:]
            continue

        joined = (buffer + "\n\n" + para).strip() if buffer else para
        if len(joined) > CHUNK_SIZE:
            chunks.append(buffer)
            # Carry the tail of the previous buffer for context continuity
            buffer = (buffer[-CHUNK_OVERLAP:] + "\n\n" + para).strip()
        else:
            buffer = joined

    if buffer.strip():
        chunks.append(buffer.strip())

    return [c for c in chunks if c.strip()]


# ── ChromaDB ─────────────────────────────────────────────────────────────────

def _get_collection(api_key: str) -> chromadb.Collection:
    _ensure_dirs()
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def collection_count(api_key: str) -> int:
    """Return total chunk count, or 0 on any error."""
    try:
        return _get_collection(api_key).count()
    except Exception:
        return 0


def search(
    query: str, api_key: str, n_results: int = 8
) -> Tuple[List[str], List[Dict]]:
    """Return (documents, metadatas) for the top-k matches.

    Returns ([], []) when the store is empty.
    Raises on ChromaDB / embedding errors so callers can surface them.
    """
    collection = _get_collection(api_key)
    count = collection.count()
    if count == 0:
        return [], []
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, count),
        include=["documents", "metadatas"],
    )
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    return docs, metas


# ── Indexing ─────────────────────────────────────────────────────────────────

def _upsert_chunks(
    collection: chromadb.Collection,
    source_key: str,
    chunks: List[str],
    extra_meta: Dict,
) -> None:
    existing = collection.get(where={"source": source_key})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    n = len(chunks)
    collection.add(
        documents=chunks,
        ids=[f"{source_key}::chunk_{i}" for i in range(n)],
        metadatas=[
            {"source": source_key, "chunk": i, **extra_meta}
            for i in range(n)
        ],
    )


def index_all_files(api_key: str) -> Tuple[int, List[str]]:
    """Embed and store every uploaded file. Returns (count, errors)."""
    collection = _get_collection(api_key)
    indexed, errors = 0, []
    for row in get_all_files():
        filename, filepath = row["filename"], row["filepath"]
        try:
            chunks = chunk_text(_extract_file_text(filepath))
            if not chunks:
                errors.append(f"{filename}: no text extracted")
                continue
            _upsert_chunks(
                collection, filename, chunks,
                {"type": "file", "filename": filename},
            )
            _mark_file_indexed(filename)
            indexed += 1
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
    return indexed, errors


def index_all_urls(api_key: str) -> Tuple[int, List[str]]:
    """Fetch, chunk, and store every tracked URL. Returns (count, errors)."""
    collection = _get_collection(api_key)
    indexed, errors = 0, []
    for row in get_all_urls():
        url = row["url"]
        max_pages = row["max_pages"] if row["max_pages"] else 1
        try:
            title, text = crawl_url(url, max_pages=max_pages)
            chunks = chunk_text(text)
            if not chunks:
                errors.append(f"{url}: no text extracted")
                continue
            _upsert_chunks(
                collection, url, chunks,
                {"type": "url", "url": url, "title": title},
            )
            _mark_url_indexed(url, title)
            indexed += 1
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    return indexed, errors


def index_all_sources(api_key: str) -> Tuple[int, List[str]]:
    """Index all files and URLs. Returns combined (count, errors)."""
    f_count, f_errors = index_all_files(api_key)
    u_count, u_errors = index_all_urls(api_key)
    return f_count + u_count, f_errors + u_errors


def reset_vector_store(api_key: str) -> None:
    """Drop the ChromaDB collection and clear all indexed flags."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    with _db() as conn:
        conn.execute("UPDATE files SET indexed=0, indexed_at=NULL")
        conn.execute("UPDATE urls  SET indexed=0, indexed_at=NULL")
