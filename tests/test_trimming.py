from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages import trim_messages as _trim_messages

from agent.memory.trimming import _ensure_user_start, _repair_tool_pairing, make_trim_middleware
from agent.prompts import SYSTEM_PROMPT


def test_system_prompt_is_str():
    assert isinstance(SYSTEM_PROMPT, str) and SYSTEM_PROMPT


def test_trim_middleware_factory():
    mw = make_trim_middleware(max_tokens=20)
    assert mw is not None


def _tc(name="x", cid="call_1"):
    return {"name": name, "args": {}, "id": cid, "type": "tool_call"}


def test_repair_removes_orphan_tool_message():
    msgs = [
        HumanMessage(content="hi"),
        ToolMessage(content="result", tool_call_id="call_orphan"),
        AIMessage(content="done"),
    ]
    repaired = _repair_tool_pairing(msgs)
    assert not any(isinstance(m, ToolMessage) for m in repaired)
    assert len(repaired) == 2


def test_repair_removes_trailing_ai_with_tool_calls():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[_tc(cid="call_1")]),
    ]
    repaired = _repair_tool_pairing(msgs)
    assert not getattr(repaired[-1], "tool_calls", None)


def test_repair_keeps_paired_tool_messages():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[_tc(cid="call_1")]),
        ToolMessage(content="result", tool_call_id="call_1"),
        AIMessage(content="final"),
    ]
    repaired = _repair_tool_pairing(msgs)
    assert len(repaired) == 4
    assert isinstance(repaired[2], ToolMessage)


def test_repair_keeps_multiple_paired_tool_calls():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[_tc("a", "c1"), _tc("b", "c2")]),
        ToolMessage(content="r1", tool_call_id="c1"),
        ToolMessage(content="r2", tool_call_id="c2"),
        AIMessage(content="final"),
    ]
    repaired = _repair_tool_pairing(msgs)
    assert len(repaired) == 5
    assert sum(1 for m in repaired if isinstance(m, ToolMessage)) == 2


def test_trim_without_start_on_keeps_non_human_messages():
    """不以 human 开头的消息序列不应被 trim 清空（曾因 start_on='human' 把消息清空导致 400）。"""
    msgs = [
        AIMessage(content="", tool_calls=[_tc(cid="c1")]),
        ToolMessage(content="result", tool_call_id="c1"),
        AIMessage(content="final answer"),
    ]
    trimmed = _trim_messages(msgs, strategy="last", token_counter="approximate", max_tokens=1000)
    repaired = _repair_tool_pairing(trimmed)
    assert len(repaired) >= 2


def test_repair_empty_returns_empty():
    """空输入返回空（空回退由 trim middleware 处理）。"""
    assert _repair_tool_pairing([]) == []


def test_ensure_user_start_adds_user_when_assistant_first():
    """trim 后若以 assistant 开头，向前补最近一个 user（glm-5.2 要求 user 开头）。"""
    human = HumanMessage(content="hi")
    ai = AIMessage(content="", tool_calls=[_tc(cid="c1")])
    tool = ToolMessage(content="r", tool_call_id="c1")
    original = [human, ai, tool]
    trimmed = [ai, tool]  # trim 裁掉了开头的 human
    result = _ensure_user_start(trimmed, original)
    assert isinstance(result[0], HumanMessage)
    assert result == [human, ai, tool]


def test_ensure_user_start_keeps_when_user_first():
    """已以 user 开头时不变。"""
    human = HumanMessage(content="hi")
    ai = AIMessage(content="reply")
    result = _ensure_user_start([human, ai], [human, ai])
    assert result == [human, ai]


def test_ensure_user_start_no_user_returns_original():
    """原消息里没有 user 时原样返回（尽力而为）。"""
    ai = AIMessage(content="reply")
    result = _ensure_user_start([ai], [ai])
    assert result == [ai]
