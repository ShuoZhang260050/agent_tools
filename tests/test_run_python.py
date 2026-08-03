from unittest.mock import patch
from agent.tools.run_python import RunPythonTool


def _run_with_ws(code, ws_path, **kwargs):
    with patch("agent.memory.user_memory.get_workspace", return_value=ws_path):
        tool = RunPythonTool()
        return tool._run(code, config={"configurable": {"user_id": 1}}, **kwargs)


def test_run_python_print(tmp_path):
    out = _run_with_ws('print("hello world")', str(tmp_path))
    assert "hello world" in out
    assert "<external_content" in out


def test_run_python_expression(tmp_path):
    out = _run_with_ws('print(2 + 3)', str(tmp_path))
    assert "5" in out


def test_run_python_import(tmp_path):
    out = _run_with_ws('import math\nprint(round(math.pi, 2))', str(tmp_path))
    assert "3.14" in out


def test_run_python_error(tmp_path):
    out = _run_with_ws('x = 1 / 0', str(tmp_path))
    assert "Error" in out or "出错" in out


def test_run_python_timeout(tmp_path):
    out = _run_with_ws('while True:\n    pass', str(tmp_path), timeout=2)
    assert "超时" in out


def test_run_python_stderr(tmp_path):
    out = _run_with_ws('print("ok")\nraise ValueError("boom")', str(tmp_path))
    assert "Traceback" in out
    assert "ValueError" in out
    assert "ok" in out
