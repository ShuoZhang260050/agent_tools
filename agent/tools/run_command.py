import subprocess

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .registry import register

_MAX_OUTPUT = 20000
_DEFAULT_TIMEOUT = 30


class RunCommandTool(BaseTool):
    name: str = "run_command"
    description: str = (
        "在工作空间中执行 shell 命令（如 git log、pytest、npm run build 等）。"
        "参数 command: 命令字符串；timeout: 超时秒数（默认 30）。"
    )

    def _run(self, command: str, timeout: int = _DEFAULT_TIMEOUT,
             config: RunnableConfig = None) -> str:
        from agent.memory.user_memory import get_workspace

        user_id = (config or {}).get("configurable", {}).get("user_id")
        if not user_id:
            return "无法执行：未识别用户身份"

        ws = get_workspace(user_id)
        if not ws:
            return "未设置工作空间，请先在页面右上角设置工作空间路径。"

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
