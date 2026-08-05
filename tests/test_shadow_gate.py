import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from agent import graph as g
from agent.config import Settings
from agent.memory.user_memory import init_tables, register_user, set_workspace
from agent.permissions import FULL_ACCESS
from agent.sandbox.shadow import get_active_workspace, clear_active_shadow
from tests.conftest import FakeToolModel


def _settings(tmp_path, monkeypatch):
    db = str(tmp_path / "shadow_gate.sqlite")
    monkeypatch.setenv("SQLITE_PATH", db)
    if os.path.exists(db):
        os.remove(db)
    return Settings(
        llm_api_key="sk-x",
        llm_model="gpt-4o-mini",
        sqlite_path=db,
        enable_tracing=False,
    )


def _setup_user(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent.memory.user_memory.get_workspace", lambda uid: str(tmp_path)
    )
    init_tables()
    user = register_user("shadowtester", "pw")
    set_workspace(user["id"], str(tmp_path))
    return user["id"]


def _build(model, settings):
    g.build_llm_with_model = lambda s, m: model
    return g.build_graph(settings)


class TestShadowGateMiddleware:
    def test_shadow_created_on_tool_call(self, tmp_path, monkeypatch):
        settings = _settings(tmp_path, monkeypatch)
        uid = _setup_user(tmp_path, monkeypatch)
        (tmp_path / "test.py").write_text("original", encoding="utf-8")

        model = FakeToolModel([
            AIMessage(content="", tool_calls=[{"name": "write_file",
                "args": {"path": "test.py", "content": "modified"}, "id": "s1", "type": "tool_call"}]),
            AIMessage(content="done"),
        ])
        graph = _build(model, settings)
        cfg = {"configurable": {"thread_id": "sg1", "user_id": uid, "permission": FULL_ACCESS}}
        graph.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)

        shadow_path = get_active_workspace(uid, "sg1")
        assert shadow_path is not None
        assert Path(shadow_path).exists()
        clear_active_shadow(uid, "sg1")

    def test_write_goes_to_shadow_not_real(self, tmp_path, monkeypatch):
        settings = _settings(tmp_path, monkeypatch)
        uid = _setup_user(tmp_path, monkeypatch)
        (tmp_path / "test.py").write_text("original", encoding="utf-8")

        model = FakeToolModel([
            AIMessage(content="", tool_calls=[{"name": "write_file",
                "args": {"path": "test.py", "content": "modified"}, "id": "s2", "type": "tool_call"}]),
            AIMessage(content="done"),
        ])
        graph = _build(model, settings)
        cfg = {"configurable": {"thread_id": "sg2", "user_id": uid, "permission": FULL_ACCESS}}
        graph.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)

        shadow_path = get_active_workspace(uid, "sg2")
        assert (Path(shadow_path) / "test.py").read_text() == "modified"
        assert (tmp_path / "test.py").read_text() == "original"
        clear_active_shadow(uid, "sg2")

    def test_read_uses_shadow(self, tmp_path, monkeypatch):
        settings = _settings(tmp_path, monkeypatch)
        uid = _setup_user(tmp_path, monkeypatch)
        (tmp_path / "read_test.py").write_text("real content", encoding="utf-8")

        model = FakeToolModel([
            AIMessage(content="", tool_calls=[{"name": "read_file",
                "args": {"path": "read_test.py"}, "id": "s3", "type": "tool_call"}]),
            AIMessage(content="done"),
        ])
        graph = _build(model, settings)
        cfg = {"configurable": {"thread_id": "sg3", "user_id": uid, "permission": FULL_ACCESS}}
        out = graph.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)

        shadow_path = get_active_workspace(uid, "sg3")
        assert (Path(shadow_path) / "read_test.py").exists()
        clear_active_shadow(uid, "sg3")

    def test_no_shadow_without_tool_calls(self, tmp_path, monkeypatch):
        settings = _settings(tmp_path, monkeypatch)
        uid = _setup_user(tmp_path, monkeypatch)

        model = FakeToolModel([
            AIMessage(content="just text, no tools"),
        ])
        graph = _build(model, settings)
        cfg = {"configurable": {"thread_id": "sg4", "user_id": uid, "permission": FULL_ACCESS}}
        graph.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)

        assert get_active_workspace(uid, "sg4") is None

    def test_run_command_uses_shadow_cwd(self, tmp_path, monkeypatch):
        settings = _settings(tmp_path, monkeypatch)
        uid = _setup_user(tmp_path, monkeypatch)
        (tmp_path / "marker.txt").write_text("real", encoding="utf-8")

        model = FakeToolModel([
            AIMessage(content="", tool_calls=[{"name": "run_command",
                "args": {"command": "echo hello > cmd_output.txt"}, "id": "s5", "type": "tool_call"}]),
            AIMessage(content="done"),
        ])
        graph = _build(model, settings)
        cfg = {"configurable": {"thread_id": "sg5", "user_id": uid, "permission": FULL_ACCESS}}
        graph.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)

        shadow_path = get_active_workspace(uid, "sg5")
        assert (Path(shadow_path) / "cmd_output.txt").exists()
        assert not (tmp_path / "cmd_output.txt").exists()
        clear_active_shadow(uid, "sg5")
