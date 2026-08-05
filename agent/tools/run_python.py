import os
import subprocess
import sys
import tempfile

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from agent.config import Settings
from .registry import register

_MAX_CODE = 10000
_MAX_OUTPUT = 20000


class RunPythonTool(BaseTool):
    name: str = "run_python"
    description: str = (
        "执行 Python 代码并返回输出。支持 print() 和表达式求值。"
        "可用于数据分析、计算、文本处理。"
        "参数 code: Python 代码字符串；timeout: 超时秒数（默认从配置读取）。"
    )

    def _run(self, code: str, timeout: int = None,
             config: RunnableConfig = None) -> str:
        from agent.memory.user_memory import get_workspace
        from agent.sandbox.shadow import get_active_workspace

        user_id = (config or {}).get("configurable", {}).get("user_id")
        if not user_id:
            return "无法执行：未识别用户身份"

        tid = (config or {}).get("configurable", {}).get("thread_id")
        ws = get_active_workspace(user_id, tid) or get_workspace(user_id)
        if not ws:
            return "未设置工作空间，请先在页面右上角设置工作空间路径。"

        if len(code) > _MAX_CODE:
            return f"代码过长（{len(code)} 字符），上限 {_MAX_CODE} 字符。"

        if timeout is None:
            timeout = Settings().run_python_timeout

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        rc = 0
        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                cwd=ws,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (result.stdout or "") + (result.stderr or "")
            rc = result.returncode
        except subprocess.TimeoutExpired:
            return f"代码执行超时（{timeout}s）"
        except Exception as e:
            return f"执行失败：{type(e).__name__}: {e}"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        output = output.strip()
        if not output:
            return "代码执行完成（无输出）。"
        if len(output) > _MAX_OUTPUT:
            output = output[:_MAX_OUTPUT] + f"\n\n[...已截断，共 {len(output)} 字符]"

        prefix = "" if rc == 0 else f"退出码 {rc}。"
        return f'<external_content source="run_python">\n{prefix}{output}\n</external_content>'


run_python = RunPythonTool()
register(run_python)
