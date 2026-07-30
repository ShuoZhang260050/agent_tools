import json
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


def _get_db():
    from agent.config import Settings
    return sqlite3.connect(Settings().sqlite_path, check_same_thread=False)


def init_traces_table():
    con = _get_db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            thread_id TEXT,
            user_id INTEGER,
            type TEXT NOT NULL,
            name TEXT,
            input_summary TEXT,
            output_summary TEXT,
            duration_ms INTEGER,
            token_usage TEXT,
            timestamp TEXT NOT NULL
        );
    """)
    con.commit()
    con.close()


def _truncate(text: str, max_len: int = 500) -> str:
    if not text:
        return ""
    return text[:max_len] + ("..." if len(text) > max_len else "")


class TracingCallbackHandler(BaseCallbackHandler):
    def __init__(self, model_name: str = ""):
        self._model_name = model_name
        self._llm_starts: dict[str, dict] = {}
        self._tool_starts: dict[str, dict] = {}

    def _extract_context(self, kwargs: dict) -> tuple[str | None, int | None]:
        metadata = kwargs.get("metadata", {}) or {}
        return metadata.get("thread_id"), metadata.get("user_id")

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._start_llm(serialized, prompts, run_id, kwargs)

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        self._start_llm(serialized, [], run_id, kwargs)

    def _start_llm(self, serialized, prompts, run_id, kwargs):
        run_id = str(run_id or uuid.uuid4())
        thread_id, user_id = self._extract_context(kwargs)
        self._llm_starts[run_id] = {
            "thread_id": thread_id,
            "user_id": user_id,
            "name": self._model_name,
            "start_time": time.time(),
            "input": prompts[0] if prompts else "",
        }

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        run_id = str(run_id)
        info = self._llm_starts.pop(run_id, None)
        if not info:
            return
        duration_ms = int((time.time() - info["start_time"]) * 1000)
        output_text = ""
        token_usage = {}
        try:
            if hasattr(response, "generations") and response.generations:
                gen = response.generations[0][0]
                output_text = gen.text if hasattr(gen, "text") else str(gen)
                msg = getattr(gen, "message", None)
                if msg:
                    usage = getattr(msg, "usage_metadata", None)
                    if usage:
                        token_usage = dict(usage)
                    else:
                        ak = getattr(msg, "additional_kwargs", None) or {}
                        tu = ak.get("token_usage") or ak.get("usage")
                        if tu:
                            token_usage = tu
            if not token_usage and hasattr(response, "llm_output") and response.llm_output:
                lo = response.llm_output
                tu = lo.get("token_usage") or lo.get("usage")
                if tu:
                    token_usage = tu
        except Exception:
            pass
        self._save_trace(
            info["thread_id"], info["user_id"], "llm", self._model_name,
            info["input"], output_text, duration_ms, token_usage,
        )

    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs):
        run_id = str(run_id or uuid.uuid4())
        thread_id, user_id = self._extract_context(kwargs)
        tool_name = ""
        if isinstance(serialized, dict):
            tool_name = serialized.get("name", "")
        self._tool_starts[run_id] = {
            "thread_id": thread_id,
            "user_id": user_id,
            "name": tool_name,
            "start_time": time.time(),
            "input": input_str,
        }

    def on_tool_end(self, output, *, run_id=None, **kwargs):
        run_id = str(run_id)
        info = self._tool_starts.pop(run_id, None)
        if not info:
            return
        duration_ms = int((time.time() - info["start_time"]) * 1000)
        output_str = str(output) if not isinstance(output, str) else output
        self._save_trace(
            info["thread_id"], info["user_id"], "tool", info["name"],
            info["input"], output_str, duration_ms, {},
        )

    def _save_trace(self, thread_id, user_id, typ, name, input_text, output_text, duration_ms, token_usage):
        con = _get_db()
        now = datetime.now().isoformat()
        trace_id = str(uuid.uuid4())
        con.execute(
            "INSERT INTO traces (trace_id, thread_id, user_id, type, name, input_summary, output_summary, duration_ms, token_usage, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (trace_id, thread_id, user_id, typ, name,
             _truncate(input_text), _truncate(output_text), duration_ms,
             json.dumps(token_usage) if token_usage else None, now),
        )
        con.commit()
        con.close()


def get_traces(thread_id: str | None = None, limit: int = 50) -> list[dict]:
    con = _get_db()
    try:
        if thread_id:
            rows = con.execute(
                "SELECT trace_id, thread_id, type, name, input_summary, output_summary, duration_ms, token_usage, timestamp "
                "FROM traces WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
                (thread_id, limit),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT trace_id, thread_id, type, name, input_summary, output_summary, duration_ms, token_usage, timestamp "
                "FROM traces ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    finally:
        con.close()
    return [
        {
            "trace_id": r[0], "thread_id": r[1], "type": r[2], "name": r[3],
            "input": r[4], "output": r[5], "duration_ms": r[6],
            "tokens": json.loads(r[7]) if r[7] else None, "timestamp": r[8],
        }
        for r in rows
    ]
