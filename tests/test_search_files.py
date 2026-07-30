from unittest.mock import patch
from agent.tools.search_files import SearchFilesTool


def _run_with_ws(query, ws_path, **kwargs):
    user_id = kwargs.pop("user_id", 1)
    with patch("agent.memory.user_memory.get_workspace", return_value=ws_path):
        tool = SearchFilesTool()
        return tool._run(query, config={"configurable": {"user_id": user_id}}, **kwargs)


def test_search_by_filename(tmp_path):
    (tmp_path / "app.py").write_text("x = 1")
    (tmp_path / "readme.md").write_text("# hello")
    out = _run_with_ws("app", str(tmp_path), search_type="filename")
    assert "app.py" in out
    assert "readme.md" not in out
    assert "<external_content" in out


def test_search_by_content(tmp_path):
    (tmp_path / "data.py").write_text("def hello():\n    return 'world'\n")
    (tmp_path / "other.py").write_text("x = 1\n")
    out = _run_with_ws("hello", str(tmp_path), search_type="content")
    assert "data.py" in out
    assert "hello" in out
    assert "other.py" not in out


def test_search_all(tmp_path):
    (tmp_path / "hello.py").write_text("print('hi')\n")
    (tmp_path / "data.py").write_text("target = 'hello world'\n")
    out = _run_with_ws("hello", str(tmp_path), search_type="all")
    assert "hello.py" in out
    assert "data.py" in out


def test_search_no_match(tmp_path):
    (tmp_path / "a.py").write_text("x = 1")
    out = _run_with_ws("nonexistent", str(tmp_path), search_type="all")
    assert "未找到" in out


def test_search_path_traversal(tmp_path):
    out = _run_with_ws("test", str(tmp_path), path="../../etc", search_type="all")
    assert "超出工作空间" in out or "不存在" in out
