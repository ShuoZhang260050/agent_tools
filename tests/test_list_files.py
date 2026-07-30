from unittest.mock import patch
from agent.tools.list_files import ListFilesTool


def _run_with_ws(path, ws_path, **kwargs):
    user_id = kwargs.pop("user_id", 1)
    with patch("agent.memory.user_memory.get_workspace", return_value=ws_path):
        tool = ListFilesTool()
        return tool._run(path, config={"configurable": {"user_id": user_id}}, **kwargs)


def test_list_files_basic(tmp_path):
    (tmp_path / "main.py").write_text("print('hi')")
    (tmp_path / "readme.md").write_text("# Readme")
    (tmp_path / "src").mkdir()
    out = _run_with_ws(".", str(tmp_path))
    assert "main.py" in out
    assert "readme.md" in out
    assert "src/" in out
    assert "<external_content" in out


def test_list_files_subdir(tmp_path):
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "app.py").write_text("x=1")
    (sub / "test.py").write_text("y=2")
    out = _run_with_ws("src", str(tmp_path))
    assert "app.py" in out
    assert "test.py" in out


def test_list_files_pattern(tmp_path):
    (tmp_path / "a.py").write_text("1")
    (tmp_path / "b.py").write_text("2")
    (tmp_path / "c.md").write_text("3")
    out = _run_with_ws(".", str(tmp_path), pattern="*.py")
    assert "a.py" in out
    assert "b.py" in out
    assert "c.md" not in out


def test_list_files_empty_dir(tmp_path):
    out = _run_with_ws(".", str(tmp_path))
    assert "为空" in out


def test_list_files_path_traversal(tmp_path):
    out = _run_with_ws("../../etc", str(tmp_path))
    assert "超出工作空间" in out or "不存在" in out
