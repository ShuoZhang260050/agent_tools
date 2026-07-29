from langchain_core.tools import tool
from duckduckgo_search import DDGS
from .registry import register

@register
@tool
def web_search(query: str) -> str:
    """搜索网络获取最新信息。输入搜索关键词。"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as e:
        return f"搜索失败：{type(e).__name__}。可能被限流或网络异常，请稍后重试。"
    if not results:
        return "无搜索结果。"
    content = "\n".join(f"- {r.get('title', '')}: {r.get('body', '')}" for r in results)
    return f'<external_content source="web_search">\n{content}\n</external_content>'
