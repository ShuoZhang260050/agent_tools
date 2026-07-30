from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .registry import register

_MAX_ENTRIES = 200


class ListFilesTool(BaseTool):
    name: str = "list_files"
    description: str = (
        "列出工作空间内的文件和子目录。"
        "参数 path: 相对路径（默认 . 为工作空间根目录）；"
        "pattern: 文件名过滤模式（如 *.py、*.md），可选。"
    )

    def _run(self, path: str = ".", pattern: str = "",
             config: RunnableConfig = None) -> str:
        from agent.memory.user_memory import get_workspace

        user_id = (config or {}).get("configurable", {}).get("user_id")
        if not user_id:
            return "无法列出：未识别用户身份"

        ws = get_workspace(user_id)
        if not ws:
            return "未设置工作空间，请先在页面右上角设置工作空间路径。"

        root = Path(ws).resolve()
        raw_path = Path(path) if path and path != "." else Path(".")
        target = raw_path.resolve() if raw_path.is_absolute() else (root / path).resolve()

        if not str(target).startswith(str(root)):
            return "拒绝访问：路径超出工作空间范围。"
        if not target.is_dir():
            return f"不是目录：{path}"

        try:
            entries = sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return f"无权限访问：{path}"

        lines = []
        count = 0
        for entry in entries:
            if count >= _MAX_ENTRIES:
                lines.append(f"...（已显示 {_MAX_ENTRIES} 项，更多未显示）")
                break
            name = entry.name
            if name.startswith("."):
                continue
            if pattern:
                if not Path(name).match(pattern):
                    continue
            if entry.is_dir():
                lines.append(f"📁 {name}/")
            else:
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1048576:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size / 1048576:.1f}MB"
                lines.append(f"📄 {name} ({size_str})")
            count += 1

        if not lines:
            return f"目录为空：{path}"

        rel = path if path and path != "." else "."
        header = f"工作空间: {rel}（{count} 项）"
        return f'<external_content source="list_files" path="{rel}">\n{header}\n' + "\n".join(lines) + "\n</external_content>"


list_files = ListFilesTool()
register(list_files)
