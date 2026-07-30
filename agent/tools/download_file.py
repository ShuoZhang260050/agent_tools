from pathlib import Path
from urllib.parse import urlparse

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from agent.memory.user_memory import add_document

_MAX_BYTES = 20_000_000
_TIMEOUT = 30
_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".json", ".log"}
_MAX_RETURN_CHARS = 20000


class DownloadFileTool(BaseTool):
    name: str = "download_file"
    description: str = (
        "从 URL 下载文件。文本类文件（txt/md/csv/json/log）返回内容，"
        "PDF 文件自动提取文本并存入当前用户的知识库。"
        "参数 url: 文件下载 URL（http/https）；filename: 保存的文件名（可选，默认从 URL 推断）。"
    )

    def _run(self, url: str, filename: str = "", config: RunnableConfig = None) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "仅支持 http 和 https URL。"
        if not parsed.netloc:
            return "无效的 URL：缺少域名。"

        if not filename:
            filename = Path(parsed.path).name or "downloaded_file"

        try:
            with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client:
                resp = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"})
                resp.raise_for_status()
                raw = resp.content[:_MAX_BYTES]
        except httpx.HTTPStatusError as e:
            return f"下载失败：HTTP {e.response.status_code}。"
        except Exception as e:
            return f"下载失败：{type(e).__name__}。可能 URL 无效或网络异常。"

        if len(resp.content) > _MAX_BYTES:
            return f"文件超过 {_MAX_BYTES // 1_000_000}MB 限制。"

        suffix = Path(filename).suffix.lower()

        if suffix in _TEXT_SUFFIXES:
            encoding = resp.encoding or "utf-8"
            text = raw.decode(encoding, errors="replace")
            if not text.strip():
                return "文件内容为空。"
            if len(text) > _MAX_RETURN_CHARS:
                text = text[:_MAX_RETURN_CHARS] + f"\n\n[...已截断，共 {len(text)} 字符，仅显示前 {_MAX_RETURN_CHARS} 字符]"
            return f'<external_content source="download_file" url="{url}">\n{text}\n</external_content>'

        if suffix == ".pdf":
            try:
                import pymupdf
            except ImportError:
                return "PDF 支持未启用：未安装 pymupdf。"
            try:
                doc = pymupdf.open(stream=raw, filetype="pdf")
                text = "\n".join(page.get_text("text") for page in doc)
                doc.close()
            except Exception as e:
                return f"PDF 解析失败：{e}。"
            if not text.strip():
                return "PDF 内容为空或无法提取文本。"

            user_id = (config or {}).get("configurable", {}).get("user_id")
            if not user_id:
                return f"PDF 文本已提取（{len(text)} 字符），但未识别用户身份，无法存入知识库。\n\n{text[:1000]}..."
            from agent.config import Settings
            from agent.memory.vectorstore import build_embeddings, ingest_document
            settings = Settings()
            embeddings = build_embeddings(settings)
            doc_id = add_document(user_id, filename, 0)
            try:
                chunk_count = ingest_document(
                    user_id, doc_id, text, filename, embeddings,
                    settings.rag_chunk_size, settings.rag_chunk_overlap,
                )
            except Exception as e:
                from agent.memory.user_memory import delete_document
                delete_document(user_id, doc_id)
                return f"PDF 已解析但存入知识库失败：{e}。"
            return f'已将 PDF "{filename}" 存入知识库（{chunk_count} 个片段，{len(text)} 字符）。可用 retrieve 工具检索。'

        return f"文件 {filename}（{len(raw)} 字节）不支持预览。支持文本类（txt/md/csv/json/log）和 PDF。"


download_file = DownloadFileTool()
