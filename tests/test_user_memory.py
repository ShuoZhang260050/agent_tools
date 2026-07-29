import os
import tempfile

from agent.memory.user_memory import init_tables, register_user, authenticate, load_memories, set_memory, get_user_threads, add_user_thread


def _setup(tmp_path):
    os.environ["SQLITE_PATH"] = str(tmp_path / "mem.sqlite")
    init_tables()
    return register_user("memuser", "pass123")


def test_register_and_authenticate(tmp_path):
    user = _setup(tmp_path)
    assert user["username"] == "memuser"
    assert authenticate("memuser", "wrong") is None
    assert authenticate("memuser", "pass123") is not None

def test_register_duplicate(tmp_path):
    _setup(tmp_path)
    import pytest
    with pytest.raises(ValueError, match="已存在"):
        register_user("memuser", "other")

def test_set_and_load_memory(tmp_path):
    user = _setup(tmp_path)
    set_memory(user["id"], "lang", "Python")
    set_memory(user["id"], "name", "Alice")
    mem = load_memories(user["id"])
    assert "lang: Python" in mem
    assert "name: Alice" in mem
    assert "<user_memory>" in mem

def test_memory_isolation(tmp_path):
    u1 = _setup(tmp_path)
    u2 = register_user("other", "pass")
    set_memory(u1["id"], "secret", "data1")
    set_memory(u2["id"], "secret", "data2")
    m1 = load_memories(u1["id"])
    m2 = load_memories(u2["id"])
    assert "data1" in m1
    assert "data2" in m2
    assert "data1" not in m2

def test_user_threads(tmp_path):
    user = _setup(tmp_path)
    add_user_thread(user["id"], "t-1")
    add_user_thread(user["id"], "t-2")
    threads = get_user_threads(user["id"])
    assert "t-1" in threads
    assert "t-2" in threads

def test_save_memory_tool():
    from agent.tools.memory_tool import save_memory
    assert save_memory.name == "save_memory"
    result = save_memory._run("k", "v", config={"configurable": {"user_id": 999}})
    assert "已记住" in result
