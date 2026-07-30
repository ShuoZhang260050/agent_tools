from unittest.mock import patch
from agent.tools.file_ops import WriteFileTool, EditFileTool


def _ws(tmp_path):
    return str(tmp_path)


def test_write_new_file(tmp_path):
    with patch("agent.memory.user_memory.get_workspace", return_value=_ws(tmp_path)):
        w = WriteFileTool()
        out = w._run("test.py", "print('hello')\n", config={"configurable": {"user_id": 1}})
    assert "已写入" in out
    assert (tmp_path / "test.py").read_text(encoding="utf-8") == "print('hello')\n"


def test_write_creates_subdirs(tmp_path):
    with patch("agent.memory.user_memory.get_workspace", return_value=_ws(tmp_path)):
        w = WriteFileTool()
        out = w._run("src/app/main.py", "x = 1\n", config={"configurable": {"user_id": 1}})
    assert "已写入" in out
    assert (tmp_path / "src" / "app" / "main.py").read_text(encoding="utf-8") == "x = 1\n"


def test_write_overwrite(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("old content", encoding="utf-8")
    with patch("agent.memory.user_memory.get_workspace", return_value=_ws(tmp_path)):
        w = WriteFileTool()
        out = w._run("data.txt", "new content", config={"configurable": {"user_id": 1}})
    assert "已写入" in out
    assert f.read_text(encoding="utf-8") == "new content"


def test_write_blocked_env(tmp_path):
    with patch("agent.memory.user_memory.get_workspace", return_value=_ws(tmp_path)):
        w = WriteFileTool()
        out = w._run(".env", "SECRET=x", config={"configurable": {"user_id": 1}})
    assert "拒绝" in out


def test_write_path_traversal(tmp_path):
    with patch("agent.memory.user_memory.get_workspace", return_value=_ws(tmp_path)):
        w = WriteFileTool()
        out = w._run("../../etc/passwd", "x", config={"configurable": {"user_id": 1}})
    assert "超出工作空间" in out


def test_edit_file_basic(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    return 1\n", encoding="utf-8")
    with patch("agent.memory.user_memory.get_workspace", return_value=_ws(tmp_path)):
        e = EditFileTool()
        out = e._run("code.py", "return 1", "return 2", config={"configurable": {"user_id": 1}})
    assert "已编辑" in out
    assert f.read_text(encoding="utf-8") == "def foo():\n    return 2\n"


def test_edit_file_not_found(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("hello world", encoding="utf-8")
    with patch("agent.memory.user_memory.get_workspace", return_value=_ws(tmp_path)):
        e = EditFileTool()
        out = e._run("code.py", "nonexistent", "x", config={"configurable": {"user_id": 1}})
    assert "未找到" in out
