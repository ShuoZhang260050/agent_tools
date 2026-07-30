from unittest.mock import patch, MagicMock
from agent.tools.download_file import download_file


def _mock_response(content: bytes, encoding: str = "utf-8"):
    resp = MagicMock()
    resp.content = content
    resp.encoding = encoding
    resp.raise_for_status = MagicMock()
    return resp


def test_download_text():
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = _mock_response(b"Hello World\nLine 2")

    with patch("agent.tools.download_file.httpx.Client", return_value=mock_client):
        out = download_file.invoke({"url": "https://example.com/data.txt"})

    assert "Hello World" in out
    assert "Line 2" in out
    assert "<external_content" in out


def test_download_pdf():
    mock_pymupdf = MagicMock()
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "PDF page text"
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_pymupdf.open.return_value = mock_doc

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = _mock_response(b"%PDF-1.4 fake")

    with patch("agent.tools.download_file.httpx.Client", return_value=mock_client), \
         patch.dict("sys.modules", {"pymupdf": mock_pymupdf}), \
         patch("agent.tools.download_file.add_document", return_value=42), \
         patch("agent.config.Settings") as mock_settings_cls, \
         patch("agent.memory.vectorstore.build_embeddings", return_value=MagicMock()), \
         patch("agent.memory.vectorstore.ingest_document", return_value=5) as mock_ingest:
        mock_settings_cls.return_value.rag_chunk_size = 500
        mock_settings_cls.return_value.rag_chunk_overlap = 50

        out = download_file.invoke(
            {"url": "https://example.com/doc.pdf"},
            config={"configurable": {"user_id": 1}},
        )

    assert "存入知识库" in out
    assert "5" in out
    mock_ingest.assert_called_once()


def test_download_bad_url():
    out = download_file.invoke({"url": "file:///etc/passwd"})
    assert "仅支持" in out


def test_download_too_large():
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    resp = _mock_response(b"x" * (21 * 1024 * 1024))
    mock_client.get.return_value = resp

    with patch("agent.tools.download_file.httpx.Client", return_value=mock_client):
        out = download_file.invoke({"url": "https://example.com/big.bin"})

    assert "超过" in out or "不支持" in out


def test_download_filename():
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = _mock_response(b"content here")

    with patch("agent.tools.download_file.httpx.Client", return_value=mock_client):
        out = download_file.invoke({"url": "https://example.com/path/to/report.json"})

    assert "content here" in out
