import json
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from agent.tools import get_tools

app = FastAPI(title="Agent API")

_graph = None


def get_graph():
    """惰性构建并缓存 agent 图（单例）。"""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class ChatReq(BaseModel):
    message: str
    thread_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tools")
def tools():
    return [{"name": t.name, "description": t.description} for t in get_tools()]


@app.post("/chat")
async def chat(req: ChatReq):
    graph = get_graph()
    tid = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    async def gen():
        async for ev in graph.astream_events(
            {"messages": [HumanMessage(content=req.message)]}, config, version="v2"
        ):
            if ev["event"] == "on_chat_model_stream":
                chunk = ev["data"]["chunk"]
                if chunk.content:
                    yield f"data: {json.dumps({'token': chunk.content}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'thread_id': tid}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
