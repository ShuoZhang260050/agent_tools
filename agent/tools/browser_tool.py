from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .registry import register

_TIMEOUT = 30
_browser = None


def _get_browser():
    global _browser
    if _browser is None:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        try:
            _browser = pw.chromium.launch(channel="msedge", headless=True)
        except Exception:
            _browser = pw.chromium.launch(headless=True)
        _browser._pw = pw
    return _browser


class BrowserTool(BaseTool):
    name: str = "browser"
    description: str = (
        "浏览器自动化工具，支持 JS 渲染的动态页面。"
        "参数 action: open（打开网页）/screenshot（截图）/click（点击）/type（输入）/extract（提取文本）/close（关闭）；"
        "url: 打开网页时的 URL；selector: CSS 选择器（点击/输入/提取时使用）；text: 输入的内容。"
    )

    def _run(self, action: str, url: str = "", selector: str = "",
             text: str = "", config: RunnableConfig = None) -> str:
        from urllib.parse import urlparse

        action = action.strip().lower()
        if action not in ("open", "screenshot", "click", "type", "extract", "close"):
            return f"无效 action：{action}。支持 open/screenshot/click/type/extract/close。"

        if action == "close":
            global _browser
            if _browser:
                try:
                    _browser.close()
                except Exception:
                    pass
                _browser = None
            return "浏览器已关闭。"

        if action == "open":
            if not url:
                return "请提供 url 参数。"
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return "仅支持 http 和 https URL。"
            try:
                browser = _get_browser()
                page = browser.new_page()
                page.goto(url, timeout=_TIMEOUT * 1000)
                page.wait_for_load_state("networkidle", timeout=_TIMEOUT * 1000)
                _page = page
            except Exception as e:
                return f"打开页面失败：{type(e).__name__}: {e}"
            global _current_page
            _current_page = page
            title = page.title()
            return f"已打开 {url}\n标题: {title}"

        try:
            page = _current_page
        except NameError:
            return "请先使用 open 打开网页。"

        if action == "screenshot":
            try:
                import base64
                png = page.screenshot()
                b64 = base64.b64encode(png).decode()
                return f'<external_content source="browser_screenshot">\n[截图已生成，{len(png)} bytes]\ndata:image/png;base64,{b64}\n</external_content>'
            except Exception as e:
                return f"截图失败：{type(e).__name__}: {e}"

        if action == "click":
            if not selector:
                return "请提供 selector 参数。"
            try:
                page.click(selector, timeout=_TIMEOUT * 1000)
                return f"已点击：{selector}"
            except Exception as e:
                return f"点击失败：{type(e).__name__}: {e}"

        if action == "type":
            if not selector:
                return "请提供 selector 参数。"
            try:
                page.fill(selector, text or "", timeout=_TIMEOUT * 1000)
                return f"已输入到 {selector}：{text}"
            except Exception as e:
                return f"输入失败：{type(e).__name__}: {e}"

        if action == "extract":
            try:
                if selector:
                    elements = page.query_selector_all(selector)
                    if not elements:
                        return f"未找到匹配 {selector} 的元素。"
                    texts = [e.inner_text() for e in elements]
                    result = "\n---\n".join(texts)
                else:
                    result = page.inner_text("body")
                if len(result) > 20000:
                    result = result[:20000] + f"\n\n[...已截断，共 {len(result)} 字符]"
                return f'<external_content source="browser_extract">\n{result}\n</external_content>'
            except Exception as e:
                return f"提取失败：{type(e).__name__}: {e}"

        return f"未知 action：{action}"


_current_page = None
browser_tool = BrowserTool()
register(browser_tool)
