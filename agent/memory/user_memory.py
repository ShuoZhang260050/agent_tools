import sqlite3
from datetime import datetime

from agent.auth import hash_password, verify_password


def _get_db():
    """获取 SQLite 数据库连接。"""
    from agent.config import Settings
    return sqlite3.connect(Settings().sqlite_path, check_same_thread=False)


def init_tables():
    """初始化所有数据库表。"""
    con = _get_db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS user_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(user_id, key)
        );
        CREATE TABLE IF NOT EXISTS user_threads (
            user_id INTEGER NOT NULL,
            thread_id TEXT NOT NULL,
            title TEXT,
            PRIMARY KEY (user_id, thread_id)
        );
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            uploaded_at TEXT NOT NULL
        );
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
    cols = {r[1] for r in con.execute("PRAGMA table_info(user_threads)").fetchall()}
    if "title" not in cols:
        con.execute("ALTER TABLE user_threads ADD COLUMN title TEXT")
    con.commit()
    con.close()


def register_user(username: str, password: str) -> dict:
    """注册新用户，返回用户信息。"""
    con = _get_db()
    existing = con.execute(
        "SELECT id FROM users WHERE username = ?", (username,)
    ).fetchone()
    if existing:
        con.close()
        raise ValueError("用户名已存在")
    now = datetime.now().isoformat()
    cur = con.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, hash_password(password), now),
    )
    con.commit()
    user_id = cur.lastrowid
    con.close()
    return {"id": user_id, "username": username}


def authenticate(username: str, password: str) -> dict | None:
    """验证用户名密码，返回用户信息或 None。"""
    con = _get_db()
    row = con.execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    con.close()
    if not row:
        return None
    if not verify_password(password, row[2]):
        return None
    return {"id": row[0], "username": row[1]}


def load_memories(user_id: int) -> str:
    """加载用户记忆，拼接为 XML 字符串。"""
    con = _get_db()
    rows = con.execute(
        "SELECT key, value FROM user_memory WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,),
    ).fetchall()
    con.close()
    if not rows:
        return ""
    lines = [f"- {k}: {v}" for k, v in rows]
    return "<user_memory>\n" + "\n".join(lines) + "\n</user_memory>"


def set_memory(user_id: int, key: str, value: str) -> str:
    """设置或更新用户记忆键值对。"""
    con = _get_db()
    now = datetime.now().isoformat()
    con.execute(
        "INSERT INTO user_memory (user_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (user_id, key, value, now),
    )
    con.commit()
    con.close()
    return f"已记住：{key} = {value}"


def get_user_threads(user_id: int) -> list[str]:
    """获取用户的所有会话 ID 列表。"""
    con = _get_db()
    rows = con.execute(
        "SELECT thread_id FROM user_threads WHERE user_id = ? ORDER BY rowid DESC",
        (user_id,),
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def get_user_threads_with_title(user_id: int) -> list[dict]:
    """获取用户会话列表（含标题）。"""
    con = _get_db()
    rows = con.execute(
        "SELECT thread_id, title FROM user_threads WHERE user_id = ? ORDER BY rowid DESC",
        (user_id,),
    ).fetchall()
    con.close()
    return [{"thread_id": r[0], "title": r[1]} for r in rows]


def add_user_thread(user_id: int, thread_id: str, title: str | None = None):
    """添加用户会话记录（如已存在则忽略）。"""
    con = _get_db()
    con.execute(
        "INSERT OR IGNORE INTO user_threads (user_id, thread_id, title) VALUES (?, ?, ?)",
        (user_id, thread_id, title),
    )
    if title is not None:
        con.execute(
            "UPDATE user_threads SET title = ? WHERE user_id = ? AND thread_id = ?",
            (title, user_id, thread_id),
        )
    con.commit()
    con.close()


def update_thread_title(user_id: int, thread_id: str, title: str):
    """更新会话标题。"""
    con = _get_db()
    con.execute(
        "UPDATE user_threads SET title = ? WHERE user_id = ? AND thread_id = ?",
        (title, user_id, thread_id),
    )
    con.commit()
    con.close()


def delete_user_thread(user_id: int, thread_id: str):
    """删除用户会话记录。"""
    con = _get_db()
    con.execute(
        "DELETE FROM user_threads WHERE user_id = ? AND thread_id = ?",
        (user_id, thread_id),
    )
    con.commit()
    con.close()


def delete_thread_data(sqlite_path: str, thread_id: str):
    """删除会话的检查点、消息(writes)与追踪记录。

    checkpoints/writes 由 LangGraph SqliteSaver 在启动时创建；
    traces 仅在 enable_tracing 时由 init_traces_table() 创建，删除前需确认存在。
    """
    con = sqlite3.connect(sqlite_path, check_same_thread=False)
    try:
        con.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        con.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        if con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='traces'"
        ).fetchone():
            con.execute("DELETE FROM traces WHERE thread_id = ?", (thread_id,))
        con.commit()
    finally:
        con.close()


def cleanup_orphan_thread_data(sqlite_path: str) -> dict:
    """清理 writes/traces 中 thread_id 不在 user_threads 的孤儿数据，返回各表删除行数。"""
    con = sqlite3.connect(sqlite_path, check_same_thread=False)
    try:
        result = {}
        for table in ("writes", "traces"):
            if not con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone():
                result[table] = 0
                continue
            cur = con.execute(
                f"DELETE FROM {table} WHERE thread_id NOT IN "
                f"(SELECT thread_id FROM user_threads)"
            )
            result[table] = cur.rowcount
        con.commit()
        return result
    finally:
        con.close()


def add_document(user_id: int, filename: str, chunk_count: int) -> int:
    """添加文档记录，返回文档 ID。"""
    con = _get_db()
    now = datetime.now().isoformat()
    cur = con.execute(
        "INSERT INTO documents (user_id, filename, chunk_count, uploaded_at) VALUES (?, ?, ?, ?)",
        (user_id, filename, chunk_count, now),
    )
    con.commit()
    doc_id = cur.lastrowid
    con.close()
    return doc_id


def list_documents(user_id: int) -> list[dict]:
    """列出用户的文档。"""
    con = _get_db()
    rows = con.execute(
        "SELECT id, filename, chunk_count, uploaded_at FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC",
        (user_id,),
    ).fetchall()
    con.close()
    return [{"id": r[0], "filename": r[1], "chunk_count": r[2], "uploaded_at": r[3]} for r in rows]


def delete_document(user_id: int, doc_id: int):
    """删除文档及其分块。"""
    con = _get_db()
    con.execute(
        "DELETE FROM document_chunks WHERE user_id = ? AND doc_id = ?",
        (user_id, doc_id),
    )
    con.execute(
        "DELETE FROM documents WHERE id = ? AND user_id = ?",
        (doc_id, user_id),
    )
    con.commit()
    con.close()


def get_workspace(user_id: int) -> str | None:
    """获取用户工作空间路径。"""
    con = _get_db()
    row = con.execute(
        "SELECT value FROM user_memory WHERE user_id = ? AND key = 'workspace_root'",
        (user_id,),
    ).fetchone()
    con.close()
    return row[0] if row else None


def set_workspace(user_id: int, path: str) -> str:
    """设置用户工作空间路径。"""
    con = _get_db()
    now = datetime.now().isoformat()
    con.execute(
        "INSERT INTO user_memory (user_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (user_id, "workspace_root", path, now),
    )
    con.commit()
    con.close()
    return path
