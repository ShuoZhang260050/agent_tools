from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .registry import register

_BLOCKED_FILES = {".env", ".env.local", ".env.production"}
_MAX_CONTENT = 500_000


def _resolve_path(file_path: str, user_id: int):
    from agent.memory.user_memory import get_workspace

    ws = get_workspace(user_id)
    if not ws:
        return None, "未设置工作空间，请先在页面右上角设置工作空间路径。"
    root = Path(ws).resolve()
    raw_path = Path(file_path) if file_path and file_path != "." else Path(".")
    path = raw_path.resolve() if raw_path.is_absolute() else (root / file_path).resolve()
    if not str(path).startswith(str(root)):
        return None, "拒绝访问：路径超出工作空间范围。"
    if path.name in _BLOCKED_FILES:
        return None, f"拒绝访问敏感文件：{path.name}"
    return path, None


class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = (
        "创建或覆盖文件。"
        "参数 path: 文件路径（相对于工作空间）；"
        "content: 文件内容。"
    )

    def _run(self, path: str, content: str,
             config: RunnableConfig = None) -> str:
        user_id = (config or {}).get("configurable", {}).get("user_id")
        if not user_id:
            return "无法写入：未识别用户身份"
        if len(content) > _MAX_CONTENT:
            return f"内容过大（{len(content)} 字符），上限 {_MAX_CONTENT} 字符。"

        p, err = _resolve_path(path, user_id)
        if err:
            return err

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except Exception as e:
            return f"写入失败：{type(e).__name__}: {e}"

        return f'已写入 {path}（{len(content)} 字符，{content.count(chr(10)) + 1} 行）'


class EditFileTool(BaseTool):
    name: str = "edit_file"
    description: str = (
        "编辑文件：将文件中第一处匹配 old_text 的内容替换为 new_text。"
        "参数 path: 文件路径；old_text: 要替换的原文（需精确匹配）；new_text: 替换后的内容。"
    )

    def _run(self, path: str, old_text: str, new_text: str,
             config: RunnableConfig = None) -> str:
        user_id = (config or {}).get("configurable", {}).get("user_id")
        if not user_id:
            return "无法编辑：未识别用户身份"

        p, err = _resolve_path(path, user_id)
        if err:
            return err
        if not p.exists():
            return f"文件不存在：{path}"
        if not p.is_file():
            return f"不是文件：{path}"

        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = p.read_text(encoding="gbk")
            except UnicodeDecodeError:
                return "文件编码无法识别（非 UTF-8/GBK）。"

        if old_text not in text:
            return f"未找到匹配的文本，请检查 old_text 是否与文件内容完全一致。"

        new_content = text.replace(old_text, new_text, 1)
        if len(new_content) > _MAX_CONTENT:
            return f"编辑后内容过大（{len(new_content)} 字符），上限 {_MAX_CONTENT} 字符。"

        try:
            p.write_text(new_content, encoding="utf-8")
        except Exception as e:
            return f"写入失败：{type(e).__name__}: {e}"

        return f'已编辑 {path}（替换了 {len(old_text)} -> {len(new_text)} 字符）'


write_file = WriteFileTool()
edit_file = EditFileTool()
register(write_file)
register(edit_file)
