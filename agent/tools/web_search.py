from langchain_core.tools import tool
from duckduckgo_search import DDGS
from .registry import register

@register
@tool
def web_search(query: str) -> str:
    """搜索网络获取最新信息。输入搜索关键词。"""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
    if not results:
        return "无搜索结果。"
    return "\n".join(f"- {r.get('title', '')}: {r.get('body', '')}" for r in results)
