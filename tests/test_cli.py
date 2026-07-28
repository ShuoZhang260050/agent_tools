from click.testing import CliRunner
from agent.cli import main


def test_tools_command():
    res = CliRunner().invoke(main, ["tools"])
    assert res.exit_code == 0
    assert "calculator" in res.output and "web_search" in res.output


def test_help():
    res = CliRunner().invoke(main, ["--help"])
    assert res.exit_code == 0
    assert "chat" in res.output and "serve" in res.output
