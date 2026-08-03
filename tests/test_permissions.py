import os

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from agent import graph as g
from agent.config import Settings
from agent.memory.user_memory import init_tables, register_user, set_workspace
from agent.permissions import (
    AUTO_APPROVE,
    FULL_ACCESS,
    REQUEST_APPROVAL,
    find_pending_approval,
    is_sensitive,
    permission_prompt_section,
)
from tests.conftest import FakeToolModel


def _settings(tmp_path, monkeypatch):
    db = str(tmp_path / "perm.sqlite")
    monkeypatch.setenv("SQLITE_PATH", db)
    if os.path.exists(db):
        os.remove(db)
    return Settings(
        llm_api_key="sk-x",
        llm_model="gpt-4o-mini",
        sqlite_path=db,
        enable_tracing=False,
    )


def _build(model, settings):
    g.build_llm_with_model = lambda s, m: model
    return g.build_graph(settings)


def _setup_user(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent.memory.user_memory.get_workspace", lambda uid: str(tmp_path)
    )
    init_tables()
    user = register_user("permtester", "pw")
    set_workspace(user["id"], str(tmp_path))
    return user["id"]


def test_is_sensitive_classification():
    assert is_sensitive("run_command")
    assert is_sensitive("write_file")
    assert is_sensitive("save_memory")
    assert not is_sensitive("calculator")
    assert not is_sensitive("web_search")


def test_permission_prompt_section_has_all_levels():
    labels = {REQUEST_APPROVAL: "请求审批", AUTO_APPROVE: "替我审批", FULL_ACCESS: "完全访问"}
    for perm, label in labels.items():
        sec = permission_prompt_section(perm)
        assert "<permission>" in sec
        assert label in sec


def test_full_access_runs_sensitive_tool_without_interrupt(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    uid = _setup_user(tmp_path, monkeypatch)
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[{"name": "run_command",
            "args": {"command": "echo full"}, "id": "f1", "type": "tool_call"}]),
        AIMessage(content="done-full"),
    ])
    graph = _build(model, settings)
    cfg = {"configurable": {"thread_id": "full", "user_id": uid, "permission": FULL_ACCESS}}
    out = graph.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)
    assert find_pending_approval(graph, cfg) is None
    types = [type(m).__name__ for m in out["messages"]]
    assert types == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert out["messages"][-1].content == "done-full"


def test_auto_approve_runs_without_interrupt(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    uid = _setup_user(tmp_path, monkeypatch)
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[{"name": "run_command",
            "args": {"command": "echo auto"}, "id": "a1", "type": "tool_call"}]),
        AIMessage(content="done-auto"),
    ])
    graph = _build(model, settings)
    cfg = {"configurable": {"thread_id": "auto", "user_id": uid, "permission": AUTO_APPROVE}}
    out = graph.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)
    assert find_pending_approval(graph, cfg) is None
    assert any(type(m).__name__ == "ToolMessage" for m in out["messages"])


def test_request_approval_interrupts_and_resumes_approved(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    uid = _setup_user(tmp_path, monkeypatch)
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[{"name": "run_command",
            "args": {"command": "echo hi"}, "id": "c1", "type": "tool_call"}]),
        AIMessage(content="resumed-ok"),
    ])
    graph = _build(model, settings)
    cfg = {"configurable": {"thread_id": "appr", "user_id": uid, "permission": REQUEST_APPROVAL}}

    out = graph.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)
    assert [type(m).__name__ for m in out["messages"]] == ["HumanMessage", "AIMessage"]
    pending = find_pending_approval(graph, cfg)
    assert pending is not None
    assert pending["tool"] == "run_command"
    assert pending["tool_call_id"] == "c1"

    out2 = graph.invoke(Command(resume="approved"), config=cfg)
    tool_msgs = [m for m in out2["messages"] if type(m).__name__ == "ToolMessage"]
    assert tool_msgs
    assert "hi" in tool_msgs[0].content
    assert out2["messages"][-1].content == "resumed-ok"
    assert find_pending_approval(graph, cfg) is None


def test_request_approval_denied_returns_cancelled_message(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    uid = _setup_user(tmp_path, monkeypatch)
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[{"name": "run_command",
            "args": {"command": "echo nope"}, "id": "d1", "type": "tool_call"}]),
        AIMessage(content="denied-ok"),
    ])
    graph = _build(model, settings)
    cfg = {"configurable": {"thread_id": "deny", "user_id": uid, "permission": REQUEST_APPROVAL}}

    graph.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)
    assert find_pending_approval(graph, cfg) is not None

    out2 = graph.invoke(Command(resume="denied"), config=cfg)
    tool_msgs = [m for m in out2["messages"] if type(m).__name__ == "ToolMessage"]
    assert tool_msgs and "取消" in tool_msgs[0].content
    assert out2["messages"][-1].content == "denied-ok"


def test_request_approval_nonsensitive_tool_runs_freely(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    uid = _setup_user(tmp_path, monkeypatch)
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[{"name": "calculator",
            "args": {"expression": "2+2"}, "id": "calc1", "type": "tool_call"}]),
        AIMessage(content="=4"),
    ])
    graph = _build(model, settings)
    cfg = {"configurable": {"thread_id": "calc", "user_id": uid, "permission": REQUEST_APPROVAL}}
    out = graph.invoke({"messages": [HumanMessage(content="算2+2")]}, config=cfg)
    assert find_pending_approval(graph, cfg) is None
    assert any(type(m).__name__ == "ToolMessage" for m in out["messages"])


def test_stream_agent_survives_interrupt_update(tmp_path, monkeypatch):
    """回归：interrupt 会在 'updates' 流里产生 __interrupt__ 节点，其 delta 是 tuple。
    _stream_agent 必须能跳过非 dict 的 delta 而不抛 AttributeError。"""
    from langgraph.types import Command

    from agent.api import _stream_agent

    settings = _settings(tmp_path, monkeypatch)
    uid = _setup_user(tmp_path, monkeypatch)
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[{"name": "run_command",
            "args": {"command": "echo hi"}, "id": "s1", "type": "tool_call"}]),
        AIMessage(content="resumed"),
    ])
    graph = _build(model, settings)
    cfg = {"configurable": {"thread_id": "stream-int", "user_id": uid,
                             "permission": REQUEST_APPROVAL}}

    first = list(_stream_agent(graph, cfg, {"messages": [HumanMessage(content="go")]},
                                REQUEST_APPROVAL))
    joined = "".join(first)
    assert "approval_request" in joined
    assert '"type": "done"' not in joined

    second = list(_stream_agent(graph, cfg, Command(resume="approved"),
                                 REQUEST_APPROVAL))
    joined2 = "".join(second)
    assert "tool_result" in joined2
    assert '"type": "done"' in joined2
