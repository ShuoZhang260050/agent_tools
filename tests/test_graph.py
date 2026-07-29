from langchain_core.messages import AIMessage
from agent.config import Settings
from agent.graph import build_graph
from tests.conftest import FakeToolModel


def _settings(tmp_path):
    return Settings(llm_api_key="sk-x", llm_model="gpt-4o-mini",
                    sqlite_path=str(tmp_path / "c.sqlite"))

def test_build_graph_compiled(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.graph.build_llm",
                        lambda s: FakeToolModel([AIMessage(content="hi")]))
    g = build_graph(_settings(tmp_path))
    assert hasattr(g, "ainvoke") and hasattr(g, "astream_events")

def test_react_loop_calls_tool(monkeypatch, tmp_path):
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[{"name": "calculator",
            "args": {"expression": "2+2"}, "id": "c1", "type": "tool_call"}]),
        AIMessage(content="结果是 4"),
    ])
    monkeypatch.setattr("agent.graph.build_llm", lambda s: model)
    g = build_graph(_settings(tmp_path))
    out = g.invoke({"messages": [{"role": "user", "content": "算 2+2"}]},
                   config={"configurable": {"thread_id": "react-test"}})
    types = [type(m).__name__ for m in out["messages"]]
    assert types == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert out["messages"][-1].content == "结果是 4"


def test_sync_stream_messages_works(monkeypatch, tmp_path):
    """回归：同步 SqliteSaver 不支持 async 方法；同步 stream(stream_mode='messages') 必须可用。"""
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[{"name": "calculator",
            "args": {"expression": "2+2"}, "id": "c1", "type": "tool_call"}]),
        AIMessage(content="结果是 4"),
    ])
    monkeypatch.setattr("agent.graph.build_llm", lambda s: model)
    g = build_graph(_settings(tmp_path))
    model_text = []
    for chunk, meta in g.stream(
        {"messages": [{"role": "user", "content": "算 2+2"}]},
        config={"configurable": {"thread_id": "stream-test"}},
        stream_mode="messages",
    ):
        if meta.get("langgraph_node") == "model" and chunk.content:
            model_text.append(chunk.content)
    assert model_text
    assert "4" in "".join(model_text)
