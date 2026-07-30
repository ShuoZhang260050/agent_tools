import io
import sys
import traceback

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .registry import register

_MAX_CODE = 10000
_TIMEOUT = 10


class RunPythonTool(BaseTool):
    name: str = "run_python"
    description: str = (
        "执行 Python 代码并返回输出。支持 print() 和表达式求值。"
        "可用于数据分析、计算、文本处理。"
        "参数 code: Python 代码字符串。"
    )

    def _run(self, code: str, config: RunnableConfig = None) -> str:
        from agent.memory.user_memory import get_workspace

        user_id = (config or {}).get("configurable", {}).get("user_id")
        if not user_id:
            return "无法执行：未识别用户身份"

        ws = get_workspace(user_id)
        if not ws:
            return "未设置工作空间，请先在页面右上角设置工作空间路径。"

        if len(code) > _MAX_CODE:
            return f"代码过长（{len(code)} 字符），上限 {_MAX_CODE} 字符。"

        old_cwd = None
        old_stdout = sys.stdout
        captured = io.StringIO()

        try:
            import os
            old_cwd = os.getcwd()
            os.chdir(ws)
            sys.stdout = captured

            glob = {"__name__": "__main__"}
            try:
                exec(compile(code, "<agent>", "exec"), glob)
            except Exception:
                sys.stdout = old_stdout
                err = traceback.format_exc()
                return f"执行出错：\n{err}"

            sys.stdout = old_stdout
            output = captured.getvalue().strip()
        except Exception as e:
            sys.stdout = old_stdout
            return f"执行失败：{type(e).__name__}: {e}"
        finally:
            if old_cwd is not None:
                os.chdir(old_cwd)
            sys.stdout = old_stdout

        if not output:
            return "代码执行完成（无输出）。"
        if len(output) > 20000:
            output = output[:20000] + f"\n\n[...已截断，共 {len(output)} 字符]"

        return f'<external_content source="run_python">\n{output}\n</external_content>'


run_python = RunPythonTool()
register(run_python)
