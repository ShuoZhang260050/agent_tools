import sqlite3
from datetime import datetime

from agent.auth import hash_password, verify_password


def _get_db():
    from agent.config import Settings
    return sqlite3.connect(Settings().sqlite_path, check_same_thread=False)


def init_tables():
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
            PRIMARY KEY (user_id, thread_id)
        );
    """)
    con.commit()
    con.close()


def register_user(username: str, password: str) -> dict:
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
    con = _get_db()
    rows = con.execute(
        "SELECT thread_id FROM user_threads WHERE user_id = ? ORDER BY rowid DESC",
        (user_id,),
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def add_user_thread(user_id: int, thread_id: str):
    con = _get_db()
    con.execute(
        "INSERT OR IGNORE INTO user_threads (user_id, thread_id) VALUES (?, ?)",
        (user_id, thread_id),
    )
    con.commit()
    con.close()
