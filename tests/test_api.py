import sqlite3

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from agent.api import app, get_current_user


def _fake_user():
    return {"id": 1, "username": "tester"}


def test_health():
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_tools():
    with TestClient(app) as c:
        names = [t["name"] for t in c.get("/tools").json()]
        assert "calculator" in names and "web_search" in names


def test_index_returns_html():
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert "Agent Chat" in r.text


def test_register_and_login(tmp_path):
    """注册 + 登录 + /me 验证。"""
    import os
    os.environ["SQLITE_PATH"] = str(tmp_path / "auth.sqlite")
    from agent.memory.user_memory import init_tables
    init_tables()
    with TestClient(app) as c:
        r = c.post("/register", json={"username": "alice", "password": "secret123"})
        assert r.status_code == 200
        assert "token" in r.json()
        r = c.post("/register", json={"username": "alice", "password": "x"})
        assert r.status_code == 409
        r = c.post("/login", json={"username": "alice", "password": "wrong"})
        assert r.status_code == 401
        r = c.post("/login", json={"username": "alice", "password": "secret123"})
        assert r.status_code == 200
        token = r.json()["token"]
        r = c.get("/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["username"] == "alice"


def test_chat_emits_events(monkeypatch):
    class Chunk:
        def __init__(self, content):
            self.content = content

    class AIMsg:
        def __init__(self, tool_calls=None, content=""):
            self.tool_calls = tool_calls
            self.content = content

    class ToolMessage:
        def __init__(self, content, tool_call_id, name):
            self.content = content
            self.tool_call_id = tool_call_id
            self.name = name

    class FakeGraph:
        def stream(self, inp, config, stream_mode=None):
            yield ("messages", (Chunk("hello"), {"langgraph_node": "model"}))
            yield ("updates", {"model": {"messages": [AIMsg(
                tool_calls=[{"id": "c1", "name": "calculator", "args": {"expression": "1+1"}}])]}})
            yield ("updates", {"tools": {"messages": [ToolMessage("2", "c1", "calculator")]}})
            yield ("messages", (Chunk(" done"), {"langgraph_node": "model"}))

    monkeypatch.setattr("agent.api.get_graph", lambda: FakeGraph())
    app.dependency_overrides[get_current_user] = _fake_user
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "hi", "thread_id": "t1"})
        assert r.status_code == 200
        body = r.text
        assert '"type": "token"' in body
        assert "tool_call" in body
        assert "tool_result" in body
        assert '"done"' in body
    app.dependency_overrides.clear()


def test_sessions(monkeypatch, tmp_path):
    db = tmp_path / "c.sqlite"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, "
        "checkpoint_id TEXT, parent_checkpoint_id TEXT, type TEXT, "
        "checkpoint BLOB, metadata BLOB)"
    )
    con.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_id) VALUES "
        "('t-a', '2024-01-02'), ('t-b', '2024-01-01')"
    )
    con.execute(
        "CREATE TABLE user_threads (user_id INTEGER, thread_id TEXT, PRIMARY KEY(user_id, thread_id))"
    )
    con.execute("INSERT INTO user_threads (user_id, thread_id) VALUES (1, 't-a'), (1, 't-b')")
    con.commit()
    con.close()

    class FakeSettings:
        sqlite_path = str(db)

    class FakeState:
        def __init__(self, msgs):
            self.values = {"messages": msgs}

    class FakeGraph:
        def get_state(self, cfg):
            tid = cfg["configurable"]["thread_id"]
            return FakeState(["m1", "m2"] if tid == "t-a" else [])

    monkeypatch.setattr("agent.api.Settings", lambda: FakeSettings())
    monkeypatch.setattr("agent.api.get_graph", lambda: FakeGraph())
    monkeypatch.setattr("agent.api.get_user_threads",
                        lambda uid: ["t-a", "t-b"])
    app.dependency_overrides[get_current_user] = _fake_user
    with TestClient(app) as c:
        r = c.get("/sessions")
        assert r.status_code == 200
        data = r.json()
        tids = [s["thread_id"] for s in data["sessions"]]
        assert tids == ["t-a", "t-b"]
        a = next(s for s in data["sessions"] if s["thread_id"] == "t-a")
        assert a["message_count"] == 2
    app.dependency_overrides.clear()


def test_get_session(monkeypatch):
    class FakeState:
        values = {"messages": [HumanMessage(content="hi"), AIMessage(content="yo")]}

    class FakeGraph:
        def get_state(self, cfg):
            return FakeState()

    monkeypatch.setattr("agent.api.get_graph", lambda: FakeGraph())
    app.dependency_overrides[get_current_user] = _fake_user
    with TestClient(app) as c:
        r = c.get("/sessions/xyz")
        assert r.status_code == 403
    app.dependency_overrides.clear()
