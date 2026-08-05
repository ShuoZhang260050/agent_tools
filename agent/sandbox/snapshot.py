import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path


def init_snapshots_table(sqlite_path: str) -> None:
    con = sqlite3.connect(sqlite_path, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS file_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            thread_id TEXT NOT NULL,
            real_path TEXT NOT NULL,
            snapshot_path TEXT NOT NULL,
            diff_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()


def save_snapshot(sqlite_path: str, user_id: int, thread_id: str,
                  real_path: str, diff: dict) -> int:
    snapshot_dir = os.path.join(
        tempfile.gettempdir(),
        f"agent_snapshot_{user_id}_{thread_id}",
        datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    )
    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)

    for rel in diff["modified"] + diff["deleted"]:
        src = Path(real_path) / rel
        if src.is_file():
            dst = Path(snapshot_dir) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    con = sqlite3.connect(sqlite_path, check_same_thread=False)
    now = datetime.now().isoformat()
    cur = con.execute(
        "INSERT INTO file_snapshots (user_id, thread_id, real_path, snapshot_path, diff_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, thread_id, real_path, snapshot_dir, json.dumps(diff), now),
    )
    con.commit()
    snapshot_id = cur.lastrowid
    con.close()
    return snapshot_id


def restore_snapshot(sqlite_path: str, snapshot_id: int) -> dict:
    con = sqlite3.connect(sqlite_path, check_same_thread=False)
    row = con.execute(
        "SELECT real_path, snapshot_path, diff_json FROM file_snapshots WHERE id = ?",
        (snapshot_id,),
    ).fetchone()
    con.close()
    if not row:
        raise ValueError(f"快照 {snapshot_id} 不存在")

    real_path, snapshot_path, diff_json = row
    diff = json.loads(diff_json)
    real = Path(real_path)
    snap = Path(snapshot_path)

    restored = 0
    for rel in diff["modified"] + diff["deleted"]:
        src = snap / rel
        if src.is_file():
            dst = real / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored += 1

    for rel in diff["added"]:
        dst = real / rel
        try:
            dst.unlink()
        except FileNotFoundError:
            pass
        restored += 1

    return {"restored": restored, "real_path": real_path}


def list_snapshots(sqlite_path: str, user_id: int, thread_id: str) -> list[dict]:
    con = sqlite3.connect(sqlite_path, check_same_thread=False)
    rows = con.execute(
        "SELECT id, real_path, diff_json, created_at FROM file_snapshots "
        "WHERE user_id = ? AND thread_id = ? ORDER BY id DESC",
        (user_id, thread_id),
    ).fetchall()
    con.close()
    return [
        {
            "id": r[0],
            "real_path": r[1],
            "diff": json.loads(r[2]),
            "created_at": r[3],
        }
        for r in rows
    ]


def get_latest_snapshot_id(sqlite_path: str, user_id: int, thread_id: str) -> int | None:
    con = sqlite3.connect(sqlite_path, check_same_thread=False)
    row = con.execute(
        "SELECT id FROM file_snapshots WHERE user_id = ? AND thread_id = ? ORDER BY id DESC LIMIT 1",
        (user_id, thread_id),
    ).fetchone()
    con.close()
    return row[0] if row else None
