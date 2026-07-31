from unittest.mock import patch, MagicMock
from agent.tools.browser_tool import BrowserTool


def test_browser_open_and_extract():
    tool = BrowserTool()
    mock_page = MagicMock()
    mock_page.title.return_value = "Test Page"
    mock_page.inner_text.return_value = "Hello World"
    with patch("agent.tools.browser_tool._browser_call") as mock_call, \
         patch("agent.tools.browser_tool._page", mock_page):
        mock_call.return_value = "已打开 https://example.com\n标题: Test Page"
        out = tool._run("open", url="https://example.com")
        assert "已打开" in out
        assert "Test Page" in out

        mock_call.return_value = '<external_content source="browser_extract">\nHello World\n</external_content>'
        out = tool._run("extract")
        assert "Hello World" in out
        assert "<external_content" in out


def test_browser_screenshot():
    tool = BrowserTool()
    with patch("agent.tools.browser_tool._browser_call") as mock_call, \
         patch("agent.tools.browser_tool._page", MagicMock()):
        mock_call.return_value = '<external_content source="browser_screenshot">\ndata:image/png;base64,abc\n</external_content>'
        out = tool._run("screenshot")
        assert "截图" in out or "base64" in out


def test_browser_click():
    tool = BrowserTool()
    with patch("agent.tools.browser_tool._browser_call") as mock_call, \
         patch("agent.tools.browser_tool._page", MagicMock()):
        mock_call.return_value = "已点击：#button"
        out = tool._run("click", selector="#button")
        assert "已点击" in out


def test_browser_type():
    tool = BrowserTool()
    with patch("agent.tools.browser_tool._browser_call") as mock_call, \
         patch("agent.tools.browser_tool._page", MagicMock()):
        mock_call.return_value = "已输入到 #input：hello"
        out = tool._run("type", selector="#input", text="hello")
        assert "已输入" in out


def test_browser_bad_scheme():
    tool = BrowserTool()
    out = tool._run("open", url="file:///etc/passwd")
    assert "仅支持" in out


def test_browser_close():
    tool = BrowserTool()
    with patch("agent.tools.browser_tool._browser_thread", None):
        out = tool._run("close")
        assert "已关闭" in out


def test_browser_no_page_open():
    tool = BrowserTool()
    with patch("agent.tools.browser_tool._page", None, create=True):
        out = tool._run("extract")
        assert "请先使用 open" in out
