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


def test_models_endpoint(monkeypatch):
    class FakeSettings:
        llm_model = "base-model"
        available_models = "base-model, vision-model"
        vision_models = "vision-model"

    monkeypatch.setattr("agent.api.Settings", lambda: FakeSettings())
    with TestClient(app) as c:
        r = c.get("/models")
        assert r.status_code == 200
        assert r.json() == {
            "models": [
                {"name": "base-model", "vision": False},
                {"name": "vision-model", "vision": True},
            ]
        }


def test_health_endpoint():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


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

    monkeypatch.setattr("agent.api.get_graph", lambda model=None: FakeGraph())
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


def test_chat_uses_requested_model(monkeypatch):
    captured = {}

    class FakeGraph:
        def stream(self, inp, config, stream_mode=None):
            if False:
                yield None

    def fake_get_graph(model=None):
        captured["model"] = model
        return FakeGraph()

    monkeypatch.setattr("agent.api.get_graph", fake_get_graph)
    monkeypatch.setattr("agent.api.add_user_thread", lambda user_id, tid, title=None: None)
    app.dependency_overrides[get_current_user] = _fake_user
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "hi", "thread_id": "t1", "model": "custom-model"})
        assert r.status_code == 200
        assert '"type": "done"' in r.text
    assert captured["model"] == "custom-model"
    app.dependency_overrides.clear()


def test_chat_with_image(monkeypatch):
    captured = {}

    class Chunk:
        def __init__(self, content):
            self.content = content

    class FakeGraph:
        def stream(self, inp, config, stream_mode=None):
            msgs = inp.get("messages", [])
            captured["message"] = msgs[0] if msgs else None
            yield ("messages", (Chunk("I see an image"), {"langgraph_node": "model"}))

    monkeypatch.setattr("agent.api.get_graph", lambda model=None: FakeGraph())
    app.dependency_overrides[get_current_user] = _fake_user
    with TestClient(app) as c:
        r = c.post("/chat", json={
            "message": "describe this",
            "thread_id": "img1",
            "model": "doubao-seed-2.0-code",
            "image": "data:image/png;base64,iVBORw0KGgo=",
        })
        assert r.status_code == 200
        assert '"type": "token"' in r.text
    msg = captured["message"]
    assert isinstance(msg.content, list)
    assert msg.content[0]["type"] == "text"
    assert msg.content[1]["type"] == "image_url"
    assert msg.content[1]["image_url"]["url"].startswith("data:image/png")
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
    monkeypatch.setattr("agent.api.get_user_threads_with_title",
                        lambda uid: [{"thread_id": "t-a", "title": None}, {"thread_id": "t-b", "title": None}])
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


def test_rename_session(monkeypatch):
    renamed = {}
    monkeypatch.setattr("agent.api.get_user_threads", lambda uid: ["t-a"])
    monkeypatch.setattr("agent.api.update_thread_title",
                        lambda uid, tid, title: renamed.update({"tid": tid, "title": title}))
    app.dependency_overrides[get_current_user] = _fake_user
    with TestClient(app) as c:
        r = c.patch("/sessions/t-a", json={"title": "My Chat"})
        assert r.status_code == 200
        assert r.json()["title"] == "My Chat"
    assert renamed == {"tid": "t-a", "title": "My Chat"}
    app.dependency_overrides.clear()


def test_delete_session(monkeypatch, tmp_path):
    db = tmp_path / "del.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_id TEXT)")
    con.execute("INSERT INTO checkpoints VALUES ('t-a', 'c1')")
    con.execute("CREATE TABLE writes (thread_id TEXT, task_id TEXT)")
    con.execute("INSERT INTO writes VALUES ('t-a', 'w1')")
    con.execute("CREATE TABLE traces (id INTEGER PRIMARY KEY, thread_id TEXT)")
    con.execute("INSERT INTO traces VALUES (1, 't-a')")
    con.commit()
    con.close()

    deleted = {}
    monkeypatch.setattr("agent.api.get_user_threads", lambda uid: ["t-a"])
    monkeypatch.setattr("agent.api.delete_user_thread",
                        lambda uid, tid: deleted.update({"tid": tid}))

    class FakeSettings:
        sqlite_path = str(db)
    monkeypatch.setattr("agent.api.Settings", lambda: FakeSettings())
    app.dependency_overrides[get_current_user] = _fake_user
    with TestClient(app) as c:
        r = c.delete("/sessions/t-a")
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"
    assert deleted == {"tid": "t-a"}
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id='t-a'").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM writes WHERE thread_id='t-a'").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM traces WHERE thread_id='t-a'").fetchone()[0] == 0
    con.close()
    app.dependency_overrides.clear()


def test_cleanup_orphan_thread_data(tmp_path):
    from agent.memory.user_memory import cleanup_orphan_thread_data

    db = tmp_path / "orphans.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE user_threads (user_id INTEGER, thread_id TEXT, PRIMARY KEY (user_id, thread_id))")
    con.execute("INSERT INTO user_threads VALUES (1, 'keep')")
    con.execute("CREATE TABLE writes (thread_id TEXT, task_id TEXT)")
    con.execute("INSERT INTO writes VALUES ('keep', 'w1')")
    con.execute("INSERT INTO writes VALUES ('orphan', 'w2')")
    con.execute("CREATE TABLE traces (id INTEGER PRIMARY KEY, thread_id TEXT)")
    con.execute("INSERT INTO traces VALUES (1, 'keep')")
    con.execute("INSERT INTO traces VALUES (2, 'orphan')")
    con.commit()
    con.close()

    result = cleanup_orphan_thread_data(str(db))

    assert result == {"writes": 1, "traces": 1}
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM writes WHERE thread_id='orphan'").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM writes WHERE thread_id='keep'").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM traces WHERE thread_id='orphan'").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM traces WHERE thread_id='keep'").fetchone()[0] == 1
    con.close()


def test_workspace_set_get(tmp_path):
    app.dependency_overrides[get_current_user] = _fake_user
    with TestClient(app) as c:
        r = c.post("/workspace", params={"path": str(tmp_path)})
        assert r.status_code == 200
        assert r.json()["workspace"] == str(tmp_path.resolve())
        r = c.get("/workspace")
        assert r.status_code == 200
        assert r.json()["workspace"] == str(tmp_path.resolve())
    app.dependency_overrides.clear()


def test_workspace_browse(tmp_path):
    (tmp_path / "subdir1").mkdir()
    (tmp_path / "subdir2").mkdir()
    (tmp_path / "file.txt").write_text("hi")
    app.dependency_overrides[get_current_user] = _fake_user
    with TestClient(app) as c:
        r = c.get("/workspace/browse", params={"path": str(tmp_path)})
        assert r.status_code == 200
        names = [d["name"] for d in r.json()["dirs"]]
        assert "subdir1" in names
        assert "subdir2" in names
        assert "file.txt" not in names
    app.dependency_overrides.clear()


def test_workspace_drives():
    app.dependency_overrides[get_current_user] = _fake_user
    with TestClient(app) as c:
        r = c.get("/workspace/drives")
        assert r.status_code == 200
        drives = r.json()["drives"]
        assert isinstance(drives, list)
        assert len(drives) > 0
    app.dependency_overrides.clear()


def test_permissions_endpoint():
    with TestClient(app) as c:
        r = c.get("/permissions")
        assert r.status_code == 200
        data = r.json()
        values = [c["value"] for c in data["choices"]]
        assert values == ["request_approval", "auto_approve", "full_access"]
        assert data["default"] == "request_approval"


def test_chat_permission_passed_to_config(monkeypatch):
    captured = {}

    class Chunk:
        def __init__(self, content):
            self.content = content

    class AIMsg:
        def __init__(self, tool_calls=None, content=""):
            self.tool_calls = tool_calls
            self.content = content

    class FakeGraph:
        def stream(self, inp, config, stream_mode=None):
            captured["permission"] = config["configurable"].get("permission")
            yield ("messages", (Chunk("ok"), {"langgraph_node": "model"}))

    monkeypatch.setattr("agent.api.get_graph", lambda model=None: FakeGraph())
    monkeypatch.setattr("agent.api.add_user_thread", lambda user_id, tid, title=None: None)
    app.dependency_overrides[get_current_user] = _fake_user
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "hi", "thread_id": "t1", "permission": "auto_approve"})
        assert r.status_code == 200
        assert '"type": "done"' in r.text
    assert captured["permission"] == "auto_approve"
    app.dependency_overrides.clear()


def test_chat_approval_request_and_resume(monkeypatch):
    """请求审批模式下：/chat 发出 approval_request，/chat/resume 完成执行。"""
    from langgraph.types import Command

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

    class Task:
        def __init__(self, interrupts):
            self.interrupts = interrupts

    class Intr:
        def __init__(self, value):
            self.value = value

    class State:
        def __init__(self, tasks):
            self.tasks = tasks

    resumed = {"v": False}

    class FakeGraph:
        def stream(self, inp, config, stream_mode=None):
            if isinstance(inp, Command):
                yield ("updates", {"tools": {"messages": [
                    ToolMessage("echo done", "c1", "run_command")]}})
                yield ("messages", (Chunk(" 完成"), {"langgraph_node": "model"}))
                resumed["v"] = True
                return
            yield ("updates", {"model": {"messages": [AIMsg(
                tool_calls=[{"id": "c1", "name": "run_command",
                             "args": {"command": "echo hi"}, "type": "tool_call"}])]}})

        def get_state(self, config):
            if resumed["v"]:
                return State(tasks=[])
            return State(tasks=[Task([Intr({
                "tool": "run_command",
                "args": {"command": "echo hi"},
                "tool_call_id": "c1",
            })])])

    monkeypatch.setattr("agent.api.get_graph", lambda model=None: FakeGraph())
    monkeypatch.setattr("agent.api.add_user_thread", lambda user_id, tid, title=None: None)
    monkeypatch.setattr("agent.api.get_user_threads", lambda uid: ["t-approve"])
    app.dependency_overrides[get_current_user] = _fake_user

    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "run echo", "thread_id": "t-approve",
                                   "permission": "request_approval"})
        assert r.status_code == 200
        body = r.text
        assert "tool_call" in body
        assert '"type": "approval_request"' in body
        assert "run_command" in body
        assert '"type": "done"' not in body

        r2 = c.post("/chat/resume", json={"thread_id": "t-approve",
                                           "decision": "approved",
                                           "permission": "request_approval"})
        assert r2.status_code == 200
        assert '"type": "tool_result"' in r2.text
        assert '"type": "done"' in r2.text
    assert resumed["v"] is True
    app.dependency_overrides.clear()
