from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from .registry import register

_MAX_CHARS = 20000
_MAX_PER_SECTION = 5000
_MAX_LINKS = 50
_MAX_BYTES = 20_000_000
_TIMEOUT = 30


def _fetch(url: str) -> tuple[str, str]:
    """Fetch URL and return (html, encoding). Raises on error."""
    with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client:
        resp = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"})
        resp.raise_for_status()
        raw = resp.content[:_MAX_BYTES]
        encoding = resp.encoding or "utf-8"
    return raw.decode(encoding, errors="replace"), encoding


def _extract_text(soup: BeautifulSoup) -> str:
    """提取页面文本。"""
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)
    if len(text) > _MAX_PER_SECTION:
        text = text[:_MAX_PER_SECTION] + f"\n[...已截断，共 {len(text)} 字符]"
    return text


def _extract_tables(soup: BeautifulSoup) -> str:
    """提取页面表格。"""
    tables = soup.find_all("table")
    if not tables:
        return "无表格。"
    parts = []
    for ti, table in enumerate(tables):
        rows = table.find_all("tr")
        if not rows:
            continue
        md_rows = []
        for row in rows:
            cells = row.find_all(["th", "td"])
            md_rows.append("| " + " | ".join(c.get_text(strip=True) for c in cells) + " |")
        if md_rows:
            header_cells = rows[0].find_all(["th", "td"])
            if header_cells:
                sep = "| " + " | ".join("---" for _ in header_cells) + " |"
                md_rows.insert(1, sep)
            parts.append(f"表 {ti + 1}:\n" + "\n".join(md_rows))
    result = "\n\n".join(parts) if parts else "无表格。"
    if len(result) > _MAX_PER_SECTION:
        result = result[:_MAX_PER_SECTION] + f"\n[...已截断，共 {len(result)} 字符]"
    return result


def _extract_links(soup: BeautifulSoup) -> str:
    """提取页面链接。"""
    seen = set()
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(strip=True) or href
        if href and href not in seen and not href.startswith("#"):
            seen.add(href)
            links.append(f"- {text}: {href}")
        if len(links) >= _MAX_LINKS:
            break
    if not links:
        return "无链接。"
    result = "\n".join(links)
    if len(result) > _MAX_PER_SECTION:
        result = result[:_MAX_PER_SECTION] + f"\n[...已截断，共 {len(links)} 条链接，仅显示前 {len(links)} 条]"
    return result


@register
@tool
def scrape_page(url: str, extract: str = "all") -> str:
    """提取网页的结构化内容。
    参数:
        url: 完整 URL（http/https）
        extract: 提取内容，可选 "text"/"tables"/"links"/"all"（默认 all）
    返回:
        - text: 页面正文
        - tables: 表格转为 Markdown 格式
        - links: 链接文本 + URL 列表"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "仅支持 http 和 https URL。"
    if not parsed.netloc:
        return "无效的 URL：缺少域名。"

    extract = extract.strip().lower()
    if extract not in ("text", "tables", "links", "all"):
        return f"无效的 extract 参数：{extract}。支持 text/tables/links/all。"

    try:
        html, _ = _fetch(url)
    except httpx.HTTPStatusError as e:
        return f"请求失败：HTTP {e.response.status_code}。"
    except Exception as e:
        return f"请求失败：{type(e).__name__}。可能 URL 无效或网络异常。"

    soup = BeautifulSoup(html, "html.parser")
    sections = []

    if extract in ("text", "all"):
        sections.append(f"=== 正文 ===\n{_extract_text(soup)}")
    if extract in ("tables", "all"):
        sections.append(f"=== 表格 ===\n{_extract_tables(soup)}")
    if extract in ("links", "all"):
        sections.append(f"=== 链接 ===\n{_extract_links(soup)}")

    result = "\n\n".join(sections)
    if len(result) > _MAX_CHARS:
        result = result[:_MAX_CHARS] + f"\n\n[...已截断，共 {len(result)} 字符，仅显示前 {_MAX_CHARS} 字符]"

    return f'<external_content source="scrape_page" url="{url}">\n{result}\n</external_content>'
