from unittest.mock import patch, MagicMock
from agent.tools.scrape_page import scrape_page


def _mock_response(html: str, encoding: str = "utf-8"):
    resp = MagicMock()
    resp.content = html.encode(encoding)
    resp.encoding = encoding
    resp.raise_for_status = MagicMock()
    return resp


_HTML_WITH_ALL = """
<html><head><title>Test</title><style>body{}</style></head>
<body><header>Nav Bar</header>
<nav><a href="/home">Home</a></nav>
<main>
  <h1>Article Title</h1>
  <p>This is the article body text.</p>
  <table>
    <tr><th>Name</th><th>Value</th></tr>
    <tr><td>A</td><td>1</td></tr>
    <tr><td>B</td><td>2</td></tr>
  </table>
  <a href="https://example.com/link1">Link One</a>
  <a href="https://example.com/link2">Link Two</a>
</main>
<footer>Copyright 2024</footer></body></html>
"""


def test_scrape_text():
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = _mock_response(_HTML_WITH_ALL)

    with patch("agent.tools.scrape_page.httpx.Client", return_value=mock_client):
        out = scrape_page.invoke({"url": "https://example.com/page", "extract": "text"})

    assert "Article Title" in out
    assert "article body text" in out
    assert "Nav Bar" not in out
    assert "Copyright" not in out
    assert "<external_content" in out


def test_scrape_tables():
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = _mock_response(_HTML_WITH_ALL)

    with patch("agent.tools.scrape_page.httpx.Client", return_value=mock_client):
        out = scrape_page.invoke({"url": "https://example.com/page", "extract": "tables"})

    assert "|" in out
    assert "Name" in out
    assert "Value" in out
    assert "---" in out


def test_scrape_links():
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = _mock_response(_HTML_WITH_ALL)

    with patch("agent.tools.scrape_page.httpx.Client", return_value=mock_client):
        out = scrape_page.invoke({"url": "https://example.com/page", "extract": "links"})

    assert "example.com/link1" in out
    assert "Link One" in out
    assert "example.com/link2" in out


def test_scrape_all():
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = _mock_response(_HTML_WITH_ALL)

    with patch("agent.tools.scrape_page.httpx.Client", return_value=mock_client):
        out = scrape_page.invoke({"url": "https://example.com/page", "extract": "all"})

    assert "正文" in out
    assert "表格" in out
    assert "链接" in out
    assert "Article Title" in out
    assert "example.com/link1" in out


def test_scrape_bad_url():
    out = scrape_page.invoke({"url": "ftp://example.com", "extract": "all"})
    assert "仅支持" in out
