from agent.memory.checkpointer import build_checkpointer
from langgraph.checkpoint.sqlite import SqliteSaver

def test_build_checkpointer_context(tmp_path):
    path = str(tmp_path / "c.sqlite")
    cm = build_checkpointer(path)
    with cm as saver:
        assert isinstance(saver, SqliteSaver)
