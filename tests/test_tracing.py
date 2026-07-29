import sqlite3

from agent.memory.tracing import TracingCallbackHandler, init_traces_table, get_traces


def _setup(tmp_path, monkeypatch):
    db = str(tmp_path / "trace.sqlite")
    monkeypatch.setenv("SQLITE_PATH", db)
    init_traces_table()
    return db


def test_traces_table_created(tmp_path, monkeypatch):
    db = _setup(tmp_path, monkeypatch)
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    con.close()
    assert "traces" in tables


def test_llm_trace_recorded(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    handler = TracingCallbackHandler()

    class FakeGen:
        text = "hello world"

    class FakeResp:
        generations = [[FakeGen()]]
        llm_output = {"token_usage": {"total_tokens": 42, "prompt_tokens": 10, "completion_tokens": 32}}

    config = {"configurable": {"thread_id": "t1", "user_id": 1}}
    handler.on_llm_start({"name": "test-model"}, ["what is 2+2"], run_id="r1", config=config)
    handler.on_llm_end(FakeResp(), run_id="r1", config=config)

    traces = get_traces(thread_id="t1")
    assert len(traces) == 1
    t = traces[0]
    assert t["type"] == "llm"
    assert t["name"] == "test-model"
    assert t["duration_ms"] >= 0
    assert t["tokens"]["total_tokens"] == 42


def test_tool_trace_recorded(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    handler = TracingCallbackHandler()

    config = {"configurable": {"thread_id": "t2", "user_id": 2}}
    handler.on_tool_start({"name": "calculator"}, '{"expression": "1+1"}', run_id="r2", config=config)
    handler.on_tool_end("2", run_id="r2", config=config)

    traces = get_traces(thread_id="t2")
    assert len(traces) == 1
    t = traces[0]
    assert t["type"] == "tool"
    assert t["name"] == "calculator"
    assert "2" in t["output"]


def test_get_traces_all(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    handler = TracingCallbackHandler()

    config = {"configurable": {"thread_id": "t1", "user_id": 1}}
    handler.on_tool_start({"name": "web_search"}, '{"query": "test"}', run_id="r3", config=config)
    handler.on_tool_end("result", run_id="r3", config=config)

    all_traces = get_traces(limit=10)
    assert len(all_traces) >= 1
