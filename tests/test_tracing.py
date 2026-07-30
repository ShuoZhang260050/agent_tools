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
    handler = TracingCallbackHandler("test-model")

    class FakeMsg:
        text = "hello world"
        usage_metadata = {"total_tokens": 42, "input_tokens": 10, "output_tokens": 32}

    class FakeGen:
        text = "hello world"
        message = FakeMsg()

    class FakeResp:
        generations = [[FakeGen()]]
        llm_output = None

    handler.on_chat_model_start({"name": "ChatOpenAI"}, [[]], run_id="r1",
                               metadata={"thread_id": "t1", "user_id": 1})
    handler.on_llm_end(FakeResp(), run_id="r1",
                       metadata={"thread_id": "t1", "user_id": 1})

    traces = get_traces(thread_id="t1")
    assert len(traces) == 1
    t = traces[0]
    assert t["type"] == "llm"
    assert t["name"] == "test-model"
    assert t["duration_ms"] >= 0
    assert t["tokens"]["total_tokens"] == 42


def test_tool_trace_recorded(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    handler = TracingCallbackHandler("test-model")

    config = {"metadata": {"thread_id": "t2", "user_id": 2}}
    handler.on_tool_start({"name": "calculator"}, '{"expression": "1+1"}', run_id="r2", **config)
    handler.on_tool_end("2", run_id="r2", **config)

    traces = get_traces(thread_id="t2")
    assert len(traces) == 1
    t = traces[0]
    assert t["type"] == "tool"
    assert t["name"] == "calculator"
    assert "2" in t["output"]


def test_get_traces_all(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    handler = TracingCallbackHandler("test-model")

    config = {"metadata": {"thread_id": "t1", "user_id": 1}}
    handler.on_tool_start({"name": "web_search"}, '{"query": "test"}', run_id="r3", **config)
    handler.on_tool_end("result", run_id="r3", **config)

    all_traces = get_traces(limit=10)
    assert len(all_traces) >= 1
