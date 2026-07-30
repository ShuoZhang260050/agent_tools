from unittest.mock import patch, MagicMock
from agent.tools.read_url import read_url


def _mock_response(html: str, encoding: str = "utf-8"):
    resp = MagicMock()
    resp.content = html.encode(encoding)
    resp.encoding = encoding
    resp.raise_for_status = MagicMock()
    return resp


def test_read_url_extracts_text():
    html = """
    <html><head><title>Test</title><style>body{}</style></head>
    <body><header>Nav</header>
    <main><h1>Hello World</h1><p>This is a test page.</p></main>
    <footer>Copyright</footer></body></html>
    """
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = _mock_response(html)

    with patch("agent.tools.read_url.httpx.Client", return_value=mock_client):
        out = read_url.invoke({"url": "https://example.com/article"})

    assert "Hello World" in out
    assert "This is a test page." in out
    assert "<external_content" in out
    assert 'source="read_url"' in out
    assert "Nav" not in out
    assert "Copyright" not in out


def test_read_url_rejects_bad_scheme():
    out = read_url.invoke({"url": "file:///etc/passwd"})
    assert "仅支持" in out

    out2 = read_url.invoke({"url": "ftp://example.com"})
    assert "仅支持" in out2


def test_read_url_handles_error():
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.side_effect = ConnectionError("boom")

    with patch("agent.tools.read_url.httpx.Client", return_value=mock_client):
        out = read_url.invoke({"url": "https://example.com/bad"})

    assert "请求失败" in out
    assert "ConnectionError" in out


def test_read_url_truncates_long():
    long_text = "A" * 30000
    html = f"<html><body><p>{long_text}</p></body></html>"
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = _mock_response(html)

    with patch("agent.tools.read_url.httpx.Client", return_value=mock_client):
        out = read_url.invoke({"url": "https://example.com/long"})

    assert "[...已截断" in out
    assert "30000" in out
