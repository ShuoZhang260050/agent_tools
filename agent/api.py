import json
import sqlite3
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage

from agent.config import Settings
from agent.graph import build_graph
from agent.tools import get_tools
from agent.auth import (
    create_access_token,
    get_current_user,
    decode_token,
)
from agent.memory.user_memory import (
    init_tables,
    register_user,
    authenticate,
    get_user_threads,
    get_user_threads_with_title,
    add_user_thread,
    update_thread_title,
    delete_user_thread,
    add_document,
    list_documents,
    delete_document,
)

app = FastAPI(title="Agent API")

_graph = None


def get_graph():
    """惰性构建并缓存 agent 图（单例）。"""
    global _graph
    if _graph is None:
        init_tables()
        _graph = build_graph()
    return _graph


_STATIC_DIR = Path(__file__).parent / "static"


def _sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _serialize_msg(m) -> dict:
    cls = type(m).__name__
    if cls == "HumanMessage":
        return {"role": "user", "content": str(m.content)}
    if cls == "AIMessage":
        d = {"role": "assistant", "content": str(m.content or "")}
        if getattr(m, "tool_calls", None):
            d["tool_calls"] = [
                {"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args")}
                for tc in m.tool_calls
            ]
        return d
    if cls == "ToolMessage":
        return {
            "role": "tool",
            "content": str(m.content),
            "name": getattr(m, "name", ""),
            "tool_call_id": getattr(m, "tool_call_id", ""),
        }
    return {"role": "unknown", "content": str(getattr(m, "content", ""))}


class ChatReq(BaseModel):
    message: str
    thread_id: str | None = None


class AuthReq(BaseModel):
    username: str
    password: str


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((_STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tools")
def tools():
    return [{"name": t.name, "description": t.description} for t in get_tools()]


@app.post("/register")
def register_endpoint(req: AuthReq):
    init_tables()
    try:
        user = register_user(req.username, req.password)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    token = create_access_token(user["id"], user["username"])
    return {"token": token, "user": user}


@app.post("/login")
def login_endpoint(req: AuthReq):
    init_tables()
    user = authenticate(req.username, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    token = create_access_token(user["id"], user["username"])
    return {"token": token, "user": user}


@app.get("/me")
def me(current: dict = Depends(get_current_user)):
    return current


@app.get("/sessions")
def list_sessions(current: dict = Depends(get_current_user)):
    """列出当前用户的会话（按最近活动排序）。"""
    settings = Settings()
    con = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
    try:
        user_threads_map = {t["thread_id"]: t["title"] for t in get_user_threads_with_title(current["id"])}
        if not user_threads_map:
            return {"sessions": []}
        tids = list(user_threads_map.keys())
        placeholders = ",".join("?" for _ in tids)
        rows = con.execute(
            f"SELECT thread_id, MAX(checkpoint_id) AS latest "
            f"FROM checkpoints WHERE thread_id IN ({placeholders}) "
            f"GROUP BY thread_id ORDER BY latest DESC",
            tids,
        ).fetchall()
    finally:
        con.close()
    graph = get_graph()
    sessions = []
    for tid, latest in rows:
        try:
            state = graph.get_state({"configurable": {"thread_id": tid}})
            n = len(state.values.get("messages", []))
        except Exception:
            n = 0
        title = user_threads_map.get(tid) or tid[:8]
        sessions.append({"thread_id": tid, "title": title, "message_count": n, "updated_at": latest})
    return {"sessions": sessions}


@app.get("/sessions/{tid}")
def get_session(tid: str, current: dict = Depends(get_current_user)):
    """获取某会话的历史消息。"""
    user_threads = get_user_threads(current["id"])
    if tid not in user_threads:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问此会话")
    graph = get_graph()
    state = graph.get_state({"configurable": {"thread_id": tid}})
    msgs = state.values.get("messages", [])
    return {"thread_id": tid, "messages": [_serialize_msg(m) for m in msgs]}


@app.post("/chat")
async def chat(req: ChatReq, current: dict = Depends(get_current_user)):
    graph = get_graph()
    tid = req.thread_id or str(uuid.uuid4())
    is_new = req.thread_id is None
    title = req.message[:30] if is_new else None
    add_user_thread(current["id"], tid, title=title)
    config = {"configurable": {"thread_id": tid, "user_id": current["id"]}}
    callbacks = getattr(graph, "_tracing_callbacks", None)
    if callbacks:
        config["callbacks"] = list(callbacks)

    def gen():
        try:
            for mode, data in graph.stream(
                {"messages": [HumanMessage(content=req.message)]},
                config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    chunk, meta = data
                    if meta.get("langgraph_node") == "model" and isinstance(chunk.content, str) and chunk.content:
                        yield _sse({"type": "token", "content": chunk.content})
                elif mode == "updates":
                    for node, delta in data.items():
                        if not delta:
                            continue
                        for m in delta.get("messages", []):
                            if node == "model" and getattr(m, "tool_calls", None):
                                for tc in m.tool_calls:
                                    yield _sse({
                                        "type": "tool_call",
                                        "id": tc.get("id"),
                                        "name": tc.get("name"),
                                        "args": tc.get("args"),
                                    })
                            elif node == "tools" and type(m).__name__ == "ToolMessage":
                                yield _sse({
                                    "type": "tool_result",
                                    "tool_call_id": getattr(m, "tool_call_id", None),
                                    "name": getattr(m, "name", ""),
                                    "content": str(m.content),
                                })
            yield _sse({"type": "done", "thread_id": tid})
        except Exception as e:
            tb = traceback.format_exc()
            try:
                with open("chat_errors.log", "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().isoformat()}] {tb}\n")
            except Exception:
                pass
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/sessions/{tid}")
def delete_session(tid: str, current: dict = Depends(get_current_user)):
    """删除指定会话。"""
    user_threads = get_user_threads(current["id"])
    if tid not in user_threads:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    delete_user_thread(current["id"], tid)
    settings = Settings()
    con = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
    try:
        con.execute("DELETE FROM checkpoints WHERE thread_id = ?", (tid,))
        con.commit()
    finally:
        con.close()
    return {"status": "deleted"}


class RenameReq(BaseModel):
    title: str


@app.patch("/sessions/{tid}")
def rename_session(tid: str, req: RenameReq, current: dict = Depends(get_current_user)):
    """重命名会话。"""
    user_threads = get_user_threads(current["id"])
    if tid not in user_threads:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    update_thread_title(current["id"], tid, req.title)
    return {"thread_id": tid, "title": req.title}


@app.get("/documents")
def list_docs(current: dict = Depends(get_current_user)):
    """列出当前用户已上传的文档。"""
    return {"documents": list_documents(current["id"])}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...), current: dict = Depends(get_current_user)):
    """上传文档到知识库（支持 .txt/.md/.pdf）。"""
    init_tables()
    filename = file.filename or "unknown.txt"
    suffix = Path(filename).suffix.lower()
    raw = await file.read()
    if suffix == ".pdf":
        try:
            import pymupdf
        except ImportError:
            raise HTTPException(status_code=400, detail="PDF 支持未启用：未安装 pymupdf")
        try:
            doc = pymupdf.open(stream=raw, filetype="pdf")
            text = "\n".join(page.get_text("text") for page in doc)
            doc.close()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF 解析失败: {e}")
    elif suffix in (".txt", ".md", ".markdown"):
        text = raw.decode("utf-8", errors="replace")
    else:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix}（支持 .txt/.md/.pdf）")
    if not text.strip():
        raise HTTPException(status_code=400, detail="文件内容为空")

    from agent.memory.vectorstore import build_embeddings, ingest_document
    settings = Settings()
    embeddings = build_embeddings(settings)
    doc_id = add_document(current["id"], filename, 0)
    try:
        chunk_count = ingest_document(
            current["id"], doc_id, text, filename, embeddings,
            settings.rag_chunk_size, settings.rag_chunk_overlap,
        )
    except Exception as e:
        delete_document(current["id"], doc_id)
        raise HTTPException(status_code=500, detail=f"文档向量化失败: {e}")
    return {"doc_id": doc_id, "filename": filename, "chunks": chunk_count}


@app.delete("/documents/{doc_id}")
def delete_doc(doc_id: int, current: dict = Depends(get_current_user)):
    """删除指定文档及其向量。"""
    docs = list_documents(current["id"])
    if not any(d["id"] == doc_id for d in docs):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    delete_document(current["id"], doc_id)
    return {"status": "deleted"}


@app.get("/traces")
def get_traces_api(thread_id: str | None = None, limit: int = 50, current: dict = Depends(get_current_user)):
    """获取调用追踪记录。"""
    from agent.memory.tracing import get_traces
    return {"traces": get_traces(thread_id=thread_id, limit=limit)}
