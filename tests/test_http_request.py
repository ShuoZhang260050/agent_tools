from unittest.mock import patch, MagicMock
from agent.tools.http_request import http_request


def _mock_response(text: str, status: int = 200, content_type: str = "application/json"):
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = MagicMock() if status < 400 else MagicMock(side_effect=Exception("HTTP error"))
    return resp


def test_http_get():
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.request.return_value = _mock_response('{"ok": true}', 200)

    with patch("agent.tools.http_request.httpx.Client", return_value=mock_client):
        out = http_request.invoke({"url": "https://api.example.com/data"})

    assert "200" in out
    assert "ok" in out
    assert "<external_content" in out
    mock_client.request.assert_called_once()
    args = mock_client.request.call_args
    assert args[0][0] == "GET"


def test_http_post():
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.request.return_value = _mock_response('{"id": 1}', 201)

    with patch("agent.tools.http_request.httpx.Client", return_value=mock_client):
        out = http_request.invoke({
            "url": "https://api.example.com/create",
            "method": "POST",
            "body": '{"name": "test"}',
        })

    assert "201" in out
    args = mock_client.request.call_args
    assert args[0][0] == "POST"


def test_http_bad_scheme():
    out = http_request.invoke({"url": "file:///etc/passwd"})
    assert "仅支持" in out


def test_http_error_status():
    import httpx
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.request.return_value = _mock_response("Not Found", 404)

    with patch("agent.tools.http_request.httpx.Client", return_value=mock_client):
        out = http_request.invoke({"url": "https://api.example.com/missing"})

    assert "404" in out


def test_http_timeout():
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.request.side_effect = TimeoutError("timed out")

    with patch("agent.tools.http_request.httpx.Client", return_value=mock_client):
        out = http_request.invoke({"url": "https://slow.example.com"})

    assert "请求失败" in out
    assert "TimeoutError" in out
