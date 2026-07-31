import queue
import threading
import uuid as _uuid
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .registry import register

_TIMEOUT = 30
_browser_thread = None
_browser = None
_page = None
_cmd_queue: queue.Queue = queue.Queue()
_result_queue: queue.Queue = queue.Queue()


def _browser_worker():
    global _browser, _page
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    try:
        _browser = pw.chromium.launch(channel="msedge", headless=True)
    except Exception:
        _browser = pw.chromium.launch(headless=True)
    _page = None
    while True:
        cmd = _cmd_queue.get()
        if cmd is None:
            break
        try:
            result = cmd()
            _result_queue.put(("ok", result))
        except Exception as e:
            _result_queue.put(("err", f"{type(e).__name__}: {e}"))
    try:
        if _browser:
            _browser.close()
    except Exception:
        pass
    pw.stop()
    _browser = None
    _page = None


def _browser_call(fn):
    global _browser_thread
    if _browser_thread is None or not _browser_thread.is_alive():
        _browser_thread = threading.Thread(target=_browser_worker, daemon=True)
        _browser_thread.start()
    _cmd_queue.put(fn)
    status, result = _result_queue.get(timeout=_TIMEOUT + 5)
    if status == "err":
        raise RuntimeError(result)
    return result


def _do_open(url):
    global _page
    page = _browser.new_page()
    page.set_viewport_size({"width": 1280, "height": 800})
    page.goto(url, timeout=_TIMEOUT * 1000)
    page.wait_for_load_state("networkidle", timeout=_TIMEOUT * 1000)
    _page = page
    return f"已打开 {url}\n标题: {page.title()}"


def _do_screenshot():
    from agent.config import Settings
    png = _page.screenshot()
    filename = f"screenshot_{_uuid.uuid4().hex[:8]}.png"
    save_dir = Path(Settings().sqlite_path).parent / "screenshots"
    save_dir.mkdir(exist_ok=True)
    (save_dir / filename).write_bytes(png)
    return f'<external_content source="browser_screenshot">截图已保存: /screenshots/{filename}</external_content>'


def _do_click(selector):
    _page.click(selector, timeout=_TIMEOUT * 1000)
    return f"已点击：{selector}"


def _do_type(selector, text):
    _page.fill(selector, text or "", timeout=_TIMEOUT * 1000)
    return f"已输入到 {selector}：{text}"


def _do_extract(selector):
    if selector:
        elements = _page.query_selector_all(selector)
        if not elements:
            return f"未找到匹配 {selector} 的元素。"
        texts = [e.inner_text() for e in elements]
        result = "\n---\n".join(texts)
    else:
        result = _page.inner_text("body")
    if len(result) > 20000:
        result = result[:20000] + f"\n\n[...已截断，共 {len(result)} 字符]"
    return f'<external_content source="browser_extract">\n{result}\n</external_content>'


class BrowserTool(BaseTool):
    name: str = "browser"
    description: str = (
        "浏览器自动化工具，支持 JS 渲染的动态页面。"
        "参数 action: open（打开网页）/screenshot（截图）/click（点击）/type（输入）/extract（提取文本）/close（关闭）；"
        "url: 打开网页时的 URL；selector: CSS 选择器（点击/输入/提取时使用）；text: 输入的内容。"
    )

    def _run(self, action: str, url: str = "", selector: str = "",
             text: str = "", config: RunnableConfig = None) -> str:
        action = action.strip().lower()
        if action not in ("open", "screenshot", "click", "type", "extract", "close"):
            return f"无效 action：{action}。支持 open/screenshot/click/type/extract/close。"

        if action == "close":
            global _browser_thread, _browser, _page
            if _browser_thread and _browser_thread.is_alive():
                _cmd_queue.put(None)
                _browser_thread.join(timeout=5)
            _browser_thread = None
            _browser = None
            _page = None
            return "浏览器已关闭。"

        if action == "open":
            if not url:
                return "请提供 url 参数。"
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return "仅支持 http 和 https URL。"
            try:
                return _browser_call(lambda: _do_open(url))
            except Exception as e:
                return f"打开页面失败：{e}"

        try:
            if _page is None:
                return "请先使用 open 打开网页。"
        except NameError:
            return "请先使用 open 打开网页。"

        try:
            if action == "screenshot":
                return _browser_call(lambda: _do_screenshot())
            elif action == "click":
                if not selector:
                    return "请提供 selector 参数。"
                return _browser_call(lambda: _do_click(selector))
            elif action == "type":
                if not selector:
                    return "请提供 selector 参数。"
                return _browser_call(lambda: _do_type(selector, text))
            elif action == "extract":
                return _browser_call(lambda: _do_extract(selector))
        except Exception as e:
            return f"{action}失败：{e}"

        return f"未知 action：{action}"


browser_tool = BrowserTool()
register(browser_tool)
