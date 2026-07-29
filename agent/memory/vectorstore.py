import json
import sqlite3
import time
from datetime import datetime

import numpy as np
from langchain_openai import OpenAIEmbeddings

from agent.config import Settings


def _get_db():
    from agent.config import Settings
    return sqlite3.connect(Settings().sqlite_path, check_same_thread=False)


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    api_key = settings.embedding_api_key
    if api_key is not None:
        api_key = api_key.get_secret_value()
    else:
        api_key = settings.llm_api_key.get_secret_value()
    base_url = settings.embedding_base_url or settings.llm_base_url
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=False,
    )


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - chunk_overlap
    return chunks


def ingest_document(
    user_id: int,
    doc_id: int,
    text: str,
    filename: str,
    embeddings: OpenAIEmbeddings,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> int:
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    if not chunks:
        return 0
    vectors = []
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        if i > 0:
            time.sleep(1)
        batch = chunks[i:i + batch_size]
        for attempt in range(4):
            try:
                vecs = embeddings.embed_documents(batch)
                vectors.extend(vecs)
                break
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    raise
    con = _get_db()
    now = datetime.now().isoformat()
    try:
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            con.execute(
                "INSERT INTO document_chunks (user_id, doc_id, chunk_index, content, embedding, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, doc_id, i, chunk, json.dumps(vec), json.dumps({"filename": filename, "chunk_index": i}), now),
            )
        con.commit()
    finally:
        con.close()
    return len(chunks)


def search(
    user_id: int,
    query: str,
    embeddings: OpenAIEmbeddings,
    top_k: int = 3,
) -> list[dict]:
    query_vec = embeddings.embed_query(query)
    con = _get_db()
    try:
        rows = con.execute(
            "SELECT id, doc_id, content, metadata, embedding FROM document_chunks WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return []
    scored = []
    for row in rows:
        chunk_id, doc_id, content, metadata_json, emb_json = row
        vec = np.array(json.loads(emb_json))
        score = float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-10))
        scored.append((score, chunk_id, doc_id, content, metadata_json))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, chunk_id, doc_id, content, metadata_json in scored[:top_k]:
        meta = json.loads(metadata_json) if metadata_json else {}
        results.append({
            "content": content,
            "score": round(score, 4),
            "doc_id": doc_id,
            "filename": meta.get("filename", ""),
            "chunk_index": meta.get("chunk_index", 0),
        })
    return results


def delete_document_chunks(user_id: int, doc_id: int) -> int:
    con = _get_db()
    try:
        cur = con.execute(
            "DELETE FROM document_chunks WHERE user_id = ? AND doc_id = ?",
            (user_id, doc_id),
        )
        con.commit()
        return cur.rowcount
    finally:
        con.close()
