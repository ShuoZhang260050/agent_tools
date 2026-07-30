from unittest.mock import patch
from agent.tools.run_command import RunCommandTool


def _run_with_ws(command, ws_path, **kwargs):
    user_id = kwargs.pop("user_id", 1)
    with patch("agent.memory.user_memory.get_workspace", return_value=ws_path):
        tool = RunCommandTool()
        return tool._run(command, config={"configurable": {"user_id": user_id}}, **kwargs)


def test_run_command_echo(tmp_path):
    out = _run_with_ws("echo hello_world", str(tmp_path))
    assert "hello_world" in out
    assert "<external_content" in out


def test_run_command_in_workspace(tmp_path):
    (tmp_path / "marker.txt").write_text("found")
    out = _run_with_ws("dir /b", str(tmp_path))
    assert "marker.txt" in out


def test_run_command_no_workspace(tmp_path):
    with patch("agent.memory.user_memory.get_workspace", return_value=None):
        tool = RunCommandTool()
        out = tool._run("echo hi", config={"configurable": {"user_id": 1}})
    assert "未设置工作空间" in out


def test_run_command_timeout():
    import subprocess
    with patch("agent.tools.run_command.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
        with patch("agent.memory.user_memory.get_workspace", return_value="C:\\"):
            tool = RunCommandTool()
            out = tool._run("sleep 999", config={"configurable": {"user_id": 1}})
    assert "超时" in out
