from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .registry import register

_MAX_FILES = 200
_MAX_MATCHES_PER_FILE = 5
_MAX_CHARS = 20000
_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".yaml", ".yml", ".xml", ".csv", ".html", ".css", ".scss",
    ".sql", ".log", ".ini", ".cfg", ".toml", ".sh", ".bat", ".ps1",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".php",
    ".vue", ".svelte",
}


class SearchFilesTool(BaseTool):
    name: str = "search_files"
    description: str = (
        "搜索工作空间中的文件。"
        "参数 query: 搜索关键词；"
        "path: 搜索起始目录（默认 . 为工作空间根目录）；"
        "search_type: filename（按文件名）、content（按内容）、all（两者都搜，默认）。"
    )

    def _run(self, query: str, path: str = ".", search_type: str = "all",
             config: RunnableConfig = None) -> str:
        from agent.memory.user_memory import get_workspace
        from agent.sandbox.shadow import get_active_workspace

        user_id = (config or {}).get("configurable", {}).get("user_id")
        if not user_id:
            return "无法搜索：未识别用户身份"

        tid = (config or {}).get("configurable", {}).get("thread_id")
        ws = get_active_workspace(user_id, tid) or get_workspace(user_id)
        if not ws:
            return "未设置工作空间，请先在页面右上角设置工作空间路径。"

        search_type = search_type.strip().lower()
        if search_type not in ("filename", "content", "all"):
            return f"无效的 search_type：{search_type}。支持 filename/content/all。"

        root = Path(ws).resolve()
        raw_path = Path(path) if path and path != "." else Path(".")
        target = raw_path.resolve() if raw_path.is_absolute() else (root / path).resolve()

        if not str(target).startswith(str(root)):
            return "拒绝访问：路径超出工作空间范围。"
        if not target.is_dir():
            return f"不是目录：{path}"

        results = []
        total_chars = 0
        file_count = 0

        for entry in sorted(target.rglob("*"), key=lambda e: (not e.is_dir(), str(e).lower())):
            if file_count >= _MAX_FILES:
                results.append(f"...（已扫描 {_MAX_FILES} 个文件，更多未显示）")
                break
            if not entry.is_file():
                continue
            if entry.name.startswith("."):
                continue
            if entry.suffix.lower() not in _TEXT_EXTS:
                continue

            file_count += 1
            matched = False

            if search_type in ("filename", "all"):
                if query.lower() in entry.name.lower():
                    rel = str(entry.relative_to(root)).replace("\\", "/")
                    results.append(f"[文件名匹配] {rel}")
                    matched = True

            if search_type in ("content", "all"):
                try:
                    text = entry.read_text(encoding="utf-8")
                except (UnicodeDecodeError, PermissionError):
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if query.lower() in line.lower():
                        if not matched:
                            rel = str(entry.relative_to(root)).replace("\\", "/")
                            results.append(f"[内容匹配] {rel}")
                        results.append(f"  {i}: {line.strip()[:200]}")
                        if sum(1 for r in results if r.startswith("  ")) >= _MAX_MATCHES_PER_FILE:
                            results.append("  ...（更多匹配已省略）")
                            break

            for r in results:
                total_chars += len(r)
            if total_chars > _MAX_CHARS:
                results.append(f"...（结果已截断，上限 {_MAX_CHARS} 字符）")
                break

        if not results or (len(results) == 1 and results[0].startswith("...")):
            return f'未找到匹配 "{query}" 的结果。'

        header = f'搜索 "{query}"（{search_type}，扫描 {file_count} 个文件）'
        return f'<external_content source="search_files">\n{header}\n' + "\n".join(results) + "\n</external_content>"


search_files = SearchFilesTool()
register(search_files)
