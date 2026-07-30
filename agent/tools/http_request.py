import json
from urllib.parse import urlparse

import httpx
from langchain_core.tools import tool
from .registry import register

_MAX_CHARS = 20000
_TIMEOUT = 30


@register
@tool
def http_request(
    url: str,
    method: str = "GET",
    headers: str = "",
    body: str = "",
) -> str:
    """发送 HTTP 请求并返回响应。
    参数:
        url: 完整 URL（http/https）
        method: GET/POST/PUT/DELETE（默认 GET）
        headers: JSON 字符串，如 '{"Authorization": "Bearer xxx", "Content-Type": "application/json"}'
        body: 请求体（POST/PUT 时使用），JSON 字符串
    返回: HTTP 状态码 + 响应正文（截断到 20000 字符）"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "仅支持 http 和 https URL。"
    if not parsed.netloc:
        return "无效的 URL：缺少域名。"

    method = method.upper().strip()
    if method not in ("GET", "POST", "PUT", "DELETE"):
        return f"不支持的 method：{method}。支持 GET/POST/PUT/DELETE。"

    hdrs = {"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"}
    if headers:
        try:
            hdrs.update(json.loads(headers))
        except json.JSONDecodeError:
            return f"headers JSON 格式错误：{headers}"

    req_body = None
    if body:
        hdrs.setdefault("Content-Type", "application/json")
        req_body = body

    try:
        with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client:
            resp = client.request(method, url, headers=hdrs, content=req_body)
    except Exception as e:
        return f"请求失败：{type(e).__name__}。可能 URL 无效或网络异常。"

    content_type = resp.headers.get("content-type", "")
    raw = resp.text
    if len(raw) > _MAX_CHARS:
        raw = raw[:_MAX_CHARS] + f"\n\n[...已截断，共 {len(raw)} 字符，仅显示前 {_MAX_CHARS} 字符]"

    return (
        f'<external_content source="http_request" url="{url}">\n'
        f"HTTP {resp.status_code} | {content_type}\n"
        f"{raw}\n"
        f"</external_content>"
    )
