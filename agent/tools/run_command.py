import subprocess

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from agent.config import Settings
from .registry import register

_MAX_OUTPUT = 20000


class RunCommandTool(BaseTool):
    """命令执行工具。"""
    name: str = "run_command"
    description: str = (
        "在工作空间中执行 shell 命令（如 git log、pytest、npm run build 等）。"
        "参数 command: 命令字符串；timeout: 超时秒数（默认从配置读取）。"
    )

    def _run(self, command: str, timeout: int = None,
             config: RunnableConfig = None) -> str:
        """在工作空间执行shell命令。"""
        from agent.memory.user_memory import get_workspace
        from agent.sandbox.shadow import get_active_workspace

        user_id = (config or {}).get("configurable", {}).get("user_id")
        if not user_id:
            return "无法执行：未识别用户身份"

        tid = (config or {}).get("configurable", {}).get("thread_id")
        ws = get_active_workspace(user_id, tid) or get_workspace(user_id)
        if not ws:
            return "未设置工作空间，请先在页面右上角设置工作空间路径。"

        if timeout is None:
            timeout = Settings().run_command_timeout

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=ws,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (result.stdout or "") + (result.stderr or "")
        except subprocess.TimeoutExpired:
            return f"命令超时（{timeout}s）：{command}"
        except Exception as e:
            return f"执行失败：{type(e).__name__}: {e}"

        output = output.strip()
        if not output:
            return f"命令执行完成（无输出）：{command}"

        if len(output) > _MAX_OUTPUT:
            output = output[:_MAX_OUTPUT] + f"\n\n[...已截断，共 {len(output)} 字符，仅显示前 {_MAX_OUTPUT} 字符]"

        return f'<external_content source="run_command" cmd="{command}">\n{output}\n</external_content>'


run_command = RunCommandTool()
register(run_command)
