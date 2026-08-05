import io
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .registry import register

_MAX_CHARS = 200_000
_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".yaml", ".yml", ".xml", ".csv", ".html", ".css", ".scss",
    ".sql", ".log", ".ini", ".cfg", ".toml", ".sh", ".bat", ".ps1",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".php",
    ".vue", ".svelte", ".docx",
}
_BLOCKED_FILES = {".env", ".env.local", ".env.production"}


class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = (
        "读取本地文件内容（代码、文档、配置）。大文件请分行读取。"
        "参数 path: 相对于工作空间的路径或绝对路径；"
        "start_line: 起始行号（默认1）；end_line: 结束行号（0=读到末尾）。"
    )

    def _run(self, path: str, start_line: int = 1, end_line: int = 0,
             config: RunnableConfig = None) -> str:
        from agent.memory.user_memory import get_workspace
        from agent.sandbox.shadow import get_active_workspace

        user_id = (config or {}).get("configurable", {}).get("user_id")
        if not user_id:
            return "无法读取：未识别用户身份"

        tid = (config or {}).get("configurable", {}).get("thread_id")
        ws = get_active_workspace(user_id, tid) or get_workspace(user_id)
        if not ws:
            return "未设置工作空间，请先在页面右上角设置工作空间路径。"

        root = Path(ws).resolve()
        raw_path = Path(path)
        resolved = raw_path.resolve() if raw_path.is_absolute() else (root / path).resolve()

        if not str(resolved).startswith(str(root)):
            return "拒绝访问：路径超出工作空间范围。"
        if not resolved.exists():
            return f"文件不存在：{path}"
        if not resolved.is_file():
            return f"不是文件：{path}"
        if resolved.name in _BLOCKED_FILES:
            return f"拒绝访问敏感文件：{resolved.name}"

        suffix = resolved.suffix.lower()
        if suffix not in _TEXT_EXTS:
            return f"不支持的文件类型：{suffix}。支持文本类文件（py/js/ts/md/json/yaml等）。"

        data = resolved.read_bytes()
        if len(data) > _MAX_CHARS * 4:
            return f"文件过大（{len(data)} 字节），上限 {_MAX_CHARS * 4} 字节。"

        if suffix == ".docx":
            try:
                import docx
                doc = docx.Document(io.BytesIO(data))
                text = "\n".join(p for p in (par.text for par in doc.paragraphs) if p.strip())
            except Exception as e:
                return f"Word 解析失败：{e}"
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = data.decode("gbk")
                except UnicodeDecodeError:
                    return "文件编码无法识别（非 UTF-8/GBK）。"

        lines = text.splitlines()
        total = len(lines)
        start = max(1, start_line)
        end = total if not end_line or end_line <= 0 else min(end_line, total)
        if start > total:
            return f"起始行 {start} 超出总行数 {total}。"

        selected = lines[start - 1:end]
        result = "\n".join(f"{i + start:4d} | {line}" for i, line in enumerate(selected))

        footer = f"[共 {total} 行，已读第 {start}-{start + len(selected) - 1} 行"
        nxt = start + len(selected)
        if nxt <= total:
            footer += f"，继续读取请用 start_line={nxt}"
        footer += "]"

        return f'<external_content source="read_file" path="{path}">\n{result}\n\n{footer}\n</external_content>'


read_file = ReadFileTool()
register(read_file)
