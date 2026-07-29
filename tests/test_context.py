from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent.prompts import SYSTEM_PROMPT
from agent.memory.state_bar import (
    _count_tool_calls,
    _build_state_bar,
    make_state_bar_middleware,
)
from agent.config import Settings


# --- Task 1: 结构化提示词 ---

def test_system_prompt_has_role():
    assert "<role>" in SYSTEM_PROMPT and "</role>" in SYSTEM_PROMPT

def test_system_prompt_has_rules():
    assert "<rules>" in SYSTEM_PROMPT and "</rules>" in SYSTEM_PROMPT

def test_system_prompt_has_sop():
    assert "<sop>" in SYSTEM_PROMPT

def test_system_prompt_has_output_format():
    assert "<output_format>" in SYSTEM_PROMPT

def test_system_prompt_has_security():
    assert "<security>" in SYSTEM_PROMPT

def test_system_prompt_no_dynamic_time():
    """提示词不应含动态信息（KV Cache 友好）。"""
    today = datetime.now().strftime("%Y-%m-%d")
    assert today not in SYSTEM_PROMPT


# --- Task 2: 状态栏 ---

def test_count_tool_calls_empty():
    assert _count_tool_calls([]) == 0

def test_count_tool_calls_with_calls():
    msgs = [
        HumanMessage(content="算 2+2"),
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "1", "type": "tool_call"}]),
        ToolMessage(content="4", tool_call_id="1"),
        AIMessage(content="结果是 4"),
    ]
    assert _count_tool_calls(msgs) == 1

def test_build_state_bar_includes_time():
    bar = _build_state_bar([])
    assert "当前时间" in bar
    today = datetime.now().strftime("%Y-%m-%d")
    assert today in bar

def test_build_state_bar_includes_counts():
    msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
    bar = _build_state_bar(msgs)
    assert "对话消息数: 2" in bar
    assert "本轮工具调用: 0 次" in bar

def test_make_state_bar_middleware_returns_middleware():
    from langchain.agents.middleware import AgentMiddleware
    mw = make_state_bar_middleware()
    assert isinstance(mw, AgentMiddleware)


# --- Task 3: 配置 ---

def test_config_new_defaults():
    s = Settings(llm_api_key="sk-test")
    assert s.model_call_limit == 25
    assert s.summary_trigger_messages == 30
    assert s.summary_keep_messages == 10
