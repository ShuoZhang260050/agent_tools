from unittest.mock import patch, MagicMock
from agent.tools.browser_tool import BrowserTool


def test_browser_open_and_extract():
    tool = BrowserTool()
    mock_page = MagicMock()
    mock_page.title.return_value = "Test Page"
    mock_page.inner_text.return_value = "Hello World"
    mock_browser = MagicMock()
    mock_browser.new_page.return_value = mock_page
    with patch("agent.tools.browser_tool._get_browser", return_value=mock_browser):
        out = tool._run("open", url="https://example.com")
        assert "已打开" in out
        assert "Test Page" in out

        with patch("agent.tools.browser_tool._current_page", mock_page):
            out = tool._run("extract")
            assert "Hello World" in out
            assert "<external_content" in out


def test_browser_screenshot():
    tool = BrowserTool()
    mock_page = MagicMock()
    mock_page.screenshot.return_value = b"\x89PNG fake"
    with patch("agent.tools.browser_tool._current_page", mock_page):
        out = tool._run("screenshot")
        assert "截图" in out or "base64" in out


def test_browser_click():
    tool = BrowserTool()
    mock_page = MagicMock()
    with patch("agent.tools.browser_tool._current_page", mock_page):
        out = tool._run("click", selector="#button")
        assert "已点击" in out
        mock_page.click.assert_called_once()


def test_browser_type():
    tool = BrowserTool()
    mock_page = MagicMock()
    with patch("agent.tools.browser_tool._current_page", mock_page):
        out = tool._run("type", selector="#input", text="hello")
        assert "已输入" in out
        mock_page.fill.assert_called_once_with("#input", "hello", timeout=30000)


def test_browser_bad_scheme():
    tool = BrowserTool()
    out = tool._run("open", url="file:///etc/passwd")
    assert "仅支持" in out


def test_browser_close():
    tool = BrowserTool()
    with patch("agent.tools.browser_tool._browser", None):
        out = tool._run("close")
        assert "已关闭" in out


def test_browser_no_page_open():
    tool = BrowserTool()
    with patch("agent.tools.browser_tool._current_page", None, create=True):
        out = tool._run("extract")
        assert "请先使用 open" in out or "NoneType" in out
