from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool


class RetrieveTool(BaseTool):
    name: str = "retrieve"
    description: str = (
        "检索用户上传的文档知识库，根据查询返回最相关的文本片段。"
        "当用户问到已上传文档相关的问题时调用此工具。"
        "参数 query: 搜索关键词或问题；top_k: 返回结果数（默认3）。"
    )

    def _run(self, query: str, top_k: int = 3, config: RunnableConfig = None) -> str:
        user_id = (config or {}).get("configurable", {}).get("user_id")
        if not user_id:
            return "无法检索：未识别用户身份"
        from agent.config import Settings
        from agent.memory.vectorstore import build_embeddings, search
        settings = Settings()
        embeddings = build_embeddings(settings)
        results = search(user_id, query, embeddings, top_k=top_k)
        if not results:
            return "知识库中暂无相关内容，或尚未上传文档。"
        parts = []
        for r in results:
            parts.append(f"[来源: {r['filename']} 片段{r['chunk_index']} 相似度: {r['score']}]\n{r['content']}")
        return f'<external_content source="knowledge_base">\n' + "\n\n---\n\n".join(parts) + "\n</external_content>"


retrieve = RetrieveTool()
