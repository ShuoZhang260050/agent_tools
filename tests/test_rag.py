import json
import sqlite3

import numpy as np

from agent.memory.vectorstore import chunk_text, ingest_document, search, delete_document_chunks


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        vec = np.zeros(8)
        for i, ch in enumerate(text[:8]):
            vec[i] = ord(ch) / 1000.0
        return vec.tolist()


def _setup_db(tmp_path):
    db = tmp_path / "rag.sqlite"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            doc_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            metadata TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    con.commit()
    con.close()
    return db


def test_chunk_text_basic():
    text = "A" * 500 + "B" * 500
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 2
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_text_overlap():
    text = "0123456789" * 100
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert chunks[1][:20] == chunks[0][-20:]


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_ingest_and_search(tmp_path, monkeypatch):
    db = _setup_db(tmp_path)
    monkeypatch.setenv("SQLITE_PATH", str(db))
    emb = FakeEmbeddings()
    text = "Python is a programming language. It is widely used for web development, data science, and AI."
    n = ingest_document(1, 1, text, "test.txt", emb, chunk_size=50, chunk_overlap=10)
    assert n > 0

    results = search(1, "Python programming", emb, top_k=2)
    assert len(results) > 0
    assert "Python" in results[0]["content"]
    assert results[0]["score"] > 0


def test_search_isolated_per_user(tmp_path, monkeypatch):
    db = _setup_db(tmp_path)
    monkeypatch.setenv("SQLITE_PATH", str(db))
    emb = FakeEmbeddings()
    ingest_document(1, 1, "user one document content", "u1.txt", emb)
    ingest_document(2, 2, "user two document content", "u2.txt", emb)
    r1 = search(1, "document", emb, top_k=5)
    r2 = search(2, "document", emb, top_k=5)
    assert all(r["filename"] == "u1.txt" for r in r1)
    assert all(r["filename"] == "u2.txt" for r in r2)


def test_delete_document_chunks(tmp_path, monkeypatch):
    db = _setup_db(tmp_path)
    monkeypatch.setenv("SQLITE_PATH", str(db))
    emb = FakeEmbeddings()
    ingest_document(1, 1, "hello world content to chunk", "t.txt", emb, chunk_size=10, chunk_overlap=2)
    deleted = delete_document_chunks(1, 1)
    assert deleted > 0
    results = search(1, "hello", emb, top_k=3)
    assert len(results) == 0


def test_search_empty_db(tmp_path, monkeypatch):
    db = _setup_db(tmp_path)
    monkeypatch.setenv("SQLITE_PATH", str(db))
    emb = FakeEmbeddings()
    results = search(1, "anything", emb, top_k=3)
    assert results == []
