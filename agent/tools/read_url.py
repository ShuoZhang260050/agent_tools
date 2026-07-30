from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from .registry import register

_MAX_CHARS = 20000
_MAX_BYTES = 20_000_000
_TIMEOUT = 30


@register
@tool
def read_url(url: str) -> str:
    """读取指定 URL 的网页正文内容。输入完整 URL（http 或 https）。
    返回纯文本，已去除 HTML 标签、脚本和样式。适用于查看网页文章、文档、API 说明等。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "仅支持 http 和 https URL。"
    if not parsed.netloc:
        return "无效的 URL：缺少域名。"

    try:
        with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client:
            resp = client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"},
            )
            resp.raise_for_status()
            raw = resp.content[:_MAX_BYTES]
            encoding = resp.encoding or "utf-8"
    except httpx.HTTPStatusError as e:
        return f"请求失败：HTTP {e.response.status_code}。"
    except Exception as e:
        return f"请求失败：{type(e).__name__}。可能 URL 无效或网络异常。"

    html = raw.decode(encoding, errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    if not text:
        return "页面内容为空或无法提取文本。"

    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + f"\n\n[...已截断，共 {len(text)} 字符，仅显示前 {_MAX_CHARS} 字符]"

    return f'<external_content source="read_url" url="{url}">\n{text}\n</external_content>'
