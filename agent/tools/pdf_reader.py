from pathlib import Path

from langchain_core.tools import tool

from .registry import register

try:
    import pymupdf  # type: ignore
except ImportError:  # 依赖缺失时给出友好提示而非崩溃
    pymupdf = None


@register
@tool
def read_pdf(
    file_path: str,
    page_start: int = 1,
    page_end: int = 0,
    max_chars: int = 1_000_000,
) -> str:
    """读取本地 PDF 文档的文本内容，用于分析、总结、检索 PDF。
    大文档请分页读取：先读前若干页，再依返回的「继续读取请用 page_start=N」提示读取后续内容。
    参数：
      file_path: PDF 文件本地路径
      page_start: 起始页码（从 1 开始，默认 1）
      page_end: 结束页码（含；0 或负数表示读到最后一页）
      max_chars: 返回的最大字符数（默认 1000000）；超出时按页边界停止并提示续读页码
    """
    if pymupdf is None:
        return "PDF 读取未启用：未安装 pymupdf，请运行 pip install pymupdf。"
    path = Path(file_path)
    if not path.exists():
        return f"文件不存在：{file_path}"
    if path.suffix.lower() != ".pdf":
        return f"不是 PDF 文件：{file_path}"

    try:
        doc = pymupdf.open(file_path)
    except Exception as e:
        return f"打开 PDF 失败：{type(e).__name__}: {e}"

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            return "PDF 已加密，需要密码才能读取。"
        total = doc.page_count
        if total == 0:
            return "PDF 无内容。"

        start = max(1, page_start)
        if start > total:
            return f"起始页 {page_start} 超出总页数 {total}。"
        end = total if not page_end or page_end <= 0 else min(page_end, total)
        if end < start:
            end = start

        parts: list[str] = []
        length = 0
        last_read = start - 1
        for pno in range(start - 1, end):
            text = doc[pno].get_text("text").rstrip()
            if not text:
                text = "（本页无可提取文本，可能是扫描件/图片页）"
            if parts and length + len(text) > max_chars:
                break
            parts.append(f"--- 第 {pno + 1} 页 ---\n{text}")
            length += len(text)
            last_read = pno + 1

        if not parts:
            first = doc[start - 1].get_text("text").rstrip()
            parts.append(f"--- 第 {start} 页 ---\n{first[:max_chars]}")
            last_read = start

        result = "\n\n".join(parts)
        footer = f"[共 {total} 页，已读第 {start}-{last_read} 页"
        nxt = last_read + 1
        if nxt <= total:
            footer += f"，继续读取请用 page_start={nxt}"
        footer += "]"
        return f'<external_content source="pdf">\n{result}\n\n{footer}\n</external_content>'
    finally:
        doc.close()
