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
    find_all_pending_approvals,
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
        AIMessage(content="", tool_calls=[{"name": "save_memory",
            "args": {"key": "test", "value": "hi"}, "id": "c1", "type": "tool_call"}]),
        AIMessage(content="resumed-ok"),
    ])
    graph = _build(model, settings)
    cfg = {"configurable": {"thread_id": "appr", "user_id": uid, "permission": REQUEST_APPROVAL}}

    out = graph.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)
    assert [type(m).__name__ for m in out["messages"]] == ["HumanMessage", "AIMessage"]
    pending = find_pending_approval(graph, cfg)
    assert pending is not None
    assert pending["tool"] == "save_memory"
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
        AIMessage(content="", tool_calls=[{"name": "save_memory",
            "args": {"key": "test", "value": "nope"}, "id": "d1", "type": "tool_call"}]),
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


def test_request_approval_sandbox_tool_no_interrupt(tmp_path, monkeypatch):
    """沙箱工具在请求审批模式下不需逐条审批，由 sync 面板统一确认。"""
    settings = _settings(tmp_path, monkeypatch)
    uid = _setup_user(tmp_path, monkeypatch)
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[{"name": "run_command",
            "args": {"command": "echo hi"}, "id": "sb1", "type": "tool_call"}]),
        AIMessage(content="done"),
    ])
    graph = _build(model, settings)
    cfg = {"configurable": {"thread_id": "sandbox", "user_id": uid, "permission": REQUEST_APPROVAL}}
    out = graph.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)
    assert find_pending_approval(graph, cfg) is None
    assert any(type(m).__name__ == "ToolMessage" for m in out["messages"])


def test_multiple_pending_interrupts_resume_all(tmp_path, monkeypatch):
    """LLM 一次返回两个敏感工具调用时，两个 interrupt 同时挂起，resume 字典恢复全部。"""
    settings = _settings(tmp_path, monkeypatch)
    uid = _setup_user(tmp_path, monkeypatch)
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[
            {"name": "save_memory",
             "args": {"key": "k1", "value": "v1"}, "id": "m1", "type": "tool_call"},
            {"name": "save_memory",
             "args": {"key": "k2", "value": "v2"}, "id": "m2", "type": "tool_call"},
        ]),
        AIMessage(content="both-done"),
    ])
    graph = _build(model, settings)
    cfg = {"configurable": {"thread_id": "multi", "user_id": uid,
                            "permission": REQUEST_APPROVAL}}

    graph.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)
    pendings = find_all_pending_approvals(graph, cfg)
    assert len(pendings) == 2
    assert all(p.get("interrupt_id") for p in pendings)

    resume_map = {p["interrupt_id"]: "approved" for p in pendings}
    out2 = graph.invoke(Command(resume=resume_map), config=cfg)
    tool_msgs = [m for m in out2["messages"] if type(m).__name__ == "ToolMessage"]
    assert len(tool_msgs) == 2
    assert out2["messages"][-1].content == "both-done"
    assert find_all_pending_approvals(graph, cfg) == []


def test_multiple_pending_interrupts_deny_all(tmp_path, monkeypatch):
    """拒绝一个等于拒绝全部：所有 interrupt 用同一 decision 恢复。"""
    settings = _settings(tmp_path, monkeypatch)
    uid = _setup_user(tmp_path, monkeypatch)
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[
            {"name": "save_memory",
             "args": {"key": "k1", "value": "v1"}, "id": "d1", "type": "tool_call"},
            {"name": "http_request",
             "args": {"url": "http://example.com", "method": "GET"}, "id": "d2", "type": "tool_call"},
        ]),
        AIMessage(content="denied-done"),
    ])
    graph = _build(model, settings)
    cfg = {"configurable": {"thread_id": "multi-deny", "user_id": uid,
                            "permission": REQUEST_APPROVAL}}

    graph.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)
    pendings = find_all_pending_approvals(graph, cfg)
    assert len(pendings) == 2

    resume_map = {p["interrupt_id"]: "denied" for p in pendings}
    out2 = graph.invoke(Command(resume=resume_map), config=cfg)
    tool_msgs = [m for m in out2["messages"] if type(m).__name__ == "ToolMessage"]
    assert len(tool_msgs) == 2
    assert all("取消" in m.content for m in tool_msgs)
    assert out2["messages"][-1].content == "denied-done"


def test_stream_agent_survives_interrupt_update(tmp_path, monkeypatch):
    """回归：interrupt 会在 'updates' 流里产生 __interrupt__ 节点，其 delta 是 tuple。
    _stream_agent 必须能跳过非 dict 的 delta 而不抛 AttributeError。"""
    from langgraph.types import Command

    from agent.api import _stream_agent

    settings = _settings(tmp_path, monkeypatch)
    uid = _setup_user(tmp_path, monkeypatch)
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[{"name": "save_memory",
            "args": {"key": "test", "value": "hi"}, "id": "s1", "type": "tool_call"}]),
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
