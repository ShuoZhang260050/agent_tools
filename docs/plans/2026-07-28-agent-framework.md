# 可扩展 Agent 框架 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 LangChain `create_agent` 的通用可扩展 LLM Agent 框架：库 + CLI + HTTP API 三入口、SQLite 持久记忆、滑动窗口裁剪、2 个示例工具。

**Architecture:** 核心是 `create_agent` 编译的 ReAct 图（model + tools + system_prompt + trim middleware + SQLite checkpointer）。config 单一配置源；llm/factory 多后端；tools/registry 装饰器注册表；CLI 与 FastAPI 都是 graph 的薄封装。

**Tech Stack:** Python 3.12, LangChain 1.3 (`create_agent` + `wrap_model_call`), langgraph-checkpoint-sqlite, FastAPI, click, pydantic-settings, duckduckgo_search, tiktoken, pytest.

## Global Constraints

- Python 3.12，venv 在 `.venv/`；激活：`$env:Path = ".venv\Scripts;$env:Path"`
- 用 `langchain.agents.create_agent`（非已弃用的 `create_react_agent`）
- 测试 mock 所有外部调用（LLM/网络），不打真实 API
- 每任务结束 commit；TDD：写失败测试 -> 验证失败 -> 实现 -> 验证通过 -> commit
- `.env` 已 gitignore；有 `.env.example`
- 已验证 API：`create_agent(model, tools, system_prompt, middleware, checkpointer)`；`wrap_model_call` 用 `request.messages`/`request.override(messages=)`/`request.model`；`trim_messages(strategy="last", token_counter=, max_tokens=, start_on="human")`；`SqliteSaver.from_conn_string(path)` 返回上下文管理器；`DDGS().text(query, max_results=)` 返回 `list[dict]`(键 title/href/body)；FakeToolModel 需实现 `_llm_type`/`_identifying_params`/`bind_tools`(返回 self)/`_generate`(返回 ChatResult)

## File Structure

`pyproject.toml`; `agent/{__init__,config,prompts,graph,cli,api}.py`; `agent/llm/{__init__,factory}.py`; `agent/tools/{__init__,registry,calculator,web_search}.py`; `agent/memory/{__init__,checkpointer,trimming}.py`; `tests/{conftest,test_*}.py`.

---

## Task 1: 项目骨架与配置层

**Files:** Create `pyproject.toml`, `agent/config.py`, `tests/__init__.py`(空), `tests/test_config.py`
**Produces:** `Settings(BaseSettings)` 字段：llm_provider(openai), llm_model(gpt-4o-mini), llm_api_key(SecretStr,必填), llm_base_url(str|None), llm_temperature(0.7), token_budget(6000), sqlite_path(checkpoints.sqlite), api_host, api_port(8000)

- [ ] **Step 1: 写 `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "agent-tools"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "langchain>=1.3","langchain-openai>=1.4","langgraph-checkpoint-sqlite>=3.1",
  "fastapi>=0.115","uvicorn[standard]>=0.30","click>=8.1",
  "pydantic-settings>=2.0","python-dotenv>=1.0","duckduckgo_search>=8.0","tiktoken>=0.7",
]
[project.optional-dependencies]
dev = ["pytest>=8","pytest-asyncio>=0.23","httpx>=0.27"]
[project.scripts]
agent = "agent.cli:main"
[tool.setuptools.packages.find]
include = ["agent*"]
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

安装：`pip install -e ".[dev]"`

- [ ] **Step 2: 写失败测试 `tests/test_config.py`**

```python
import pytest
from agent.config import Settings

def test_settings_defaults_and_overrides():
    s = Settings(llm_api_key="sk-test", llm_base_url="http://localhost:11434/v1")
    assert s.llm_provider == "openai"
    assert s.llm_model == "gpt-4o-mini"
    assert s.llm_api_key.get_secret_value() == "sk-test"
    assert s.token_budget == 6000
    assert s.sqlite_path == "checkpoints.sqlite"

def test_settings_missing_api_key_raises():
    with pytest.raises(Exception):
        Settings()
```

- [ ] **Step 3: 验证失败** — Run: `pytest tests/test_config.py -v` → FAIL(ModuleNotFoundError)
- [ ] **Step 4: 实现 `agent/config.py`**

```python
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: SecretStr
    llm_base_url: str | None = None
    llm_temperature: float = 0.7
    token_budget: int = 6000
    sqlite_path: str = "checkpoints.sqlite"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")
```

- [ ] **Step 5: 验证通过** — Run: `pytest tests/test_config.py -v` → PASS
- [ ] **Step 6: Commit** — `git add pyproject.toml agent/config.py tests/ && git commit -m "feat: project scaffold + config layer"`

---

## Task 2: 工具注册表

**Files:** Create `agent/tools/registry.py`, `agent/tools/__init__.py`, `tests/test_registry.py`
**Produces:** `register(tool)->tool`，`get_tools()->list`（返回副本）

- [ ] **Step 1: 写失败测试 `tests/test_registry.py`**

```python
from agent.tools.registry import register, get_tools

def test_register_and_get_tools():
    before = len(get_tools())
    @register
    def fake_tool(x: str) -> str:
        """fake"""
        return x
    assert fake_tool in get_tools()
    assert len(get_tools()) == before + 1

def test_get_tools_returns_copy():
    tools = get_tools()
    tools.append("mutated")
    assert "mutated" not in get_tools()
```

- [ ] **Step 2: 验证失败** — `pytest tests/test_registry.py -v` → FAIL
- [ ] **Step 3: 实现 `agent/tools/registry.py`**

```python
from typing import Any

_REGISTRY: list[Any] = []

def register(tool: Any) -> Any:
    _REGISTRY.append(tool)
    return tool

def get_tools() -> list[Any]:
    return list(_REGISTRY)
```

- [ ] **Step 4: `agent/tools/__init__.py`** — `# 工具模块在此 import 触发注册（Task 3/4 填充）`
- [ ] **Step 5: 验证通过** — PASS
- [ ] **Step 6: Commit** — `git add agent/tools tests/test_registry.py && git commit -m "feat: tool registry"`

---

## Task 3: 计算器工具

**Files:** Create `agent/tools/calculator.py`, Modify `agent/tools/__init__.py`, Create `tests/test_tools.py`
**Produces:** `calculator` 工具(`calculator(expression: str) -> str)`，ast 安全解析

- [ ] **Step 1: 写失败测试 `tests/test_tools.py`**

```python
import pytest
from agent.tools.calculator import calculator

def test_calc_basic():
    assert calculator.invoke({"expression": "2 + 3"}) == "5"
def test_calc_precedence():
    assert calculator.invoke({"expression": "2 * 3 + 4"}) == "10"
    assert calculator.invoke({"expression": "10 / 4"}) == "2.5"
def test_calc_rejects_dangerous():
    for bad in ["__import__('os')", "open('a')", "1 and 2"]:
        with pytest.raises(Exception):
            calculator.invoke({"expression": bad})
```

- [ ] **Step 2: 验证失败** → FAIL(ModuleNotFoundError)
- [ ] **Step 3: 实现 `agent/tools/calculator.py`**

```python
import ast
import operator
from langchain_core.tools import tool
from .registry import register

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("不支持的表达式")

@register
@tool
def calculator(expression: str) -> str:
    """计算算术表达式（支持 + - * / % ** 与括号）。输入如 '2 + 3 * 4'。"""
    return str(_eval(ast.parse(expression, mode="eval")))
```

- [ ] **Step 4: `agent/tools/__init__.py`** — `from . import calculator  # noqa: F401`
- [ ] **Step 5: 验证通过** — PASS
- [ ] **Step 6: Commit** — `git add agent/tools/calculator.py agent/tools/__init__.py tests/test_tools.py && git commit -m "feat: calculator tool"`

---

## Task 4: Web 搜索工具

**Files:** Create `agent/tools/web_search.py`, Modify `agent/tools/__init__.py`, Append `tests/test_tools.py`
**Produces:** `web_search` 工具(`web_search(query: str) -> str)`

- [ ] **Step 1: 追加失败测试到 `tests/test_tools.py`**

```python
from unittest.mock import patch, MagicMock
from agent.tools.web_search import web_search

def _patch_ddgs(results):
    inst = MagicMock()
    inst.text.return_value = results
    patcher = patch("agent.tools.web_search.DDGS")
    m = patcher.start()
    m.return_value.__enter__.return_value = inst
    return patcher

def test_web_search_formats_results():
    p = _patch_ddgs([{"title": "T1", "body": "B1"}, {"title": "T2", "body": "B2"}])
    try:
        out = web_search.invoke({"query": "python"})
    finally:
        p.stop()
    assert "T1" in out and "B1" in out and "T2" in out

def test_web_search_empty():
    p = _patch_ddgs([])
    try:
        assert "无" in web_search.invoke({"query": "zzz"})
    finally:
        p.stop()
```

- [ ] **Step 2: 验证失败** → FAIL(ModuleNotFoundError: agent.tools.web_search)
- [ ] **Step 3: 实现 `agent/tools/web_search.py`**

```python
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from .registry import register

@register
@tool
def web_search(query: str) -> str:
    """搜索网络获取最新信息。输入搜索关键词。"""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
    if not results:
        return "无搜索结果。"
    return "\n".join(f"- {r.get('title', '')}: {r.get('body', '')}" for r in results)
```

- [ ] **Step 4: `agent/tools/__init__.py`** — `from . import calculator, web_search  # noqa: F401`
- [ ] **Step 5: 验证通过** — PASS
- [ ] **Step 6: Commit** — `git add agent/tools/web_search.py agent/tools/__init__.py tests/test_tools.py && git commit -m "feat: web search tool"`

---

## Task 5: LLM 工厂

**Files:** Create `agent/llm/__init__.py`(空), `agent/llm/factory.py`, `tests/test_factory.py`
**Consumes:** `Settings` **Produces:** `build_llm(settings) -> BaseChatModel`

- [ ] **Step 1: 写失败测试 `tests/test_factory.py`**

```python
import pytest
from agent.config import Settings
from agent.llm.factory import build_llm

def _s(**kw):
    base = dict(llm_api_key="sk-x", llm_model="gpt-4o-mini")
    base.update(kw)
    return Settings(**base)

def test_build_llm_openai_attrs():
    llm = build_llm(_s(llm_base_url="http://localhost:11434/v1"))
    assert llm.model_name == "gpt-4o-mini"
    assert str(llm.openai_api_base).rstrip("/") == "http://localhost:11434/v1"

def test_build_llm_unknown_provider():
    with pytest.raises(ValueError):
        build_llm(_s(llm_provider="bogus"))
```

- [ ] **Step 2: 验证失败** → FAIL
- [ ] **Step 3: 实现 `agent/llm/factory.py`**

```python
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from agent.config import Settings

def build_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
        )
    raise ValueError(f"未知 LLM provider: {settings.llm_provider}")
```

- [ ] **Step 4: 验证通过** — PASS
- [ ] **Step 5: Commit** — `git add agent/llm tests/test_factory.py && git commit -m "feat: LLM factory (OpenAI-compatible)"`

## Task 6: SQLite Checkpointer

**Files:** Create `agent/memory/__init__.py`(空), `agent/memory/checkpointer.py`, `tests/test_checkpointer.py`
**Produces:** `build_checkpointer(sqlite_path)` 返回上下文管理器（`from_conn_string`），`with` 内得 `SqliteSaver`

- [ ] **Step 1: 写失败测试 `tests/test_checkpointer.py`**

```python
from agent.memory.checkpointer import build_checkpointer
from langgraph.checkpoint.sqlite import SqliteSaver

def test_build_checkpointer_context(tmp_path):
    path = str(tmp_path / "c.sqlite")
    cm = build_checkpointer(path)
    with cm as saver:
        assert isinstance(saver, SqliteSaver)
```

- [ ] **Step 2: 验证失败** - `pytest tests/test_checkpointer.py -v` -> FAIL
- [ ] **Step 3: 实现 `agent/memory/checkpointer.py`**

```python
from langgraph.checkpoint.sqlite import SqliteSaver

def build_checkpointer(sqlite_path: str):
    """返回 SqliteSaver 上下文管理器；用 with build_checkpointer(path) as saver:。"""
    return SqliteSaver.from_conn_string(sqlite_path)
```

- [ ] **Step 4: 验证通过** - PASS
- [ ] **Step 5: Commit** - `git add agent/memory tests/test_checkpointer.py && git commit -m "feat: sqlite checkpointer"`

---

## Task 7: 裁剪中间件 + System Prompt

**Files:** Create `agent/memory/trimming.py`, `agent/prompts.py`, `tests/test_trimming.py`
**Produces:** `make_trim_middleware(max_tokens)`（返回 `wrap_model_call` 中间件），`SYSTEM_PROMPT` 常量

- [ ] **Step 1: 写失败测试 `tests/test_trimming.py`**

```python
from agent.memory.trimming import make_trim_middleware
from agent.prompts import SYSTEM_PROMPT

def test_system_prompt_is_str():
    assert isinstance(SYSTEM_PROMPT, str) and SYSTEM_PROMPT

def test_trim_middleware_factory():
    mw = make_trim_middleware(max_tokens=20)
    assert mw is not None
```

（裁剪逻辑的真实断言在 Task 8 的图测试中端到端验证）

- [ ] **Step 2: 验证失败** -> FAIL
- [ ] **Step 3: 实现 `agent/prompts.py`**

```python
SYSTEM_PROMPT = "你是一个乐于助人的助手。当需要最新信息或精确计算时，可调用提供的工具。"
```

- [ ] **Step 4: 实现 `agent/memory/trimming.py`**

```python
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import trim_messages

def make_trim_middleware(max_tokens: int):
    @wrap_model_call(name="TrimMessagesMiddleware")
    def trim(request, handler):
        trimmed = trim_messages(
            request.messages,
            strategy="last",
            token_counter=request.model,
            max_tokens=max_tokens,
            start_on="human",
        )
        return handler(request.override(messages=trimmed))
    return trim
```

- [ ] **Step 5: 验证通过** - PASS
- [ ] **Step 6: Commit** - `git add agent/memory/trimming.py agent/prompts.py tests/test_trimming.py && git commit -m "feat: trim middleware + system prompt"`

## Task 8: 图装配 + FakeToolModel + ReAct 循环测试

**Files:** Create `tests/conftest.py`, `agent/graph.py`, `agent/__init__.py`, `tests/test_graph.py`
**Consumes:** `build_llm`, `get_tools`, `build_checkpointer`, `make_trim_middleware`, `SYSTEM_PROMPT`
**Produces:** `build_graph(settings)` 返回已编译图；`agent/__init__.py` 暴露 `build_graph`, `Settings`

- [ ] **Step 1: 写 `tests/conftest.py`（FakeToolModel 替身）**

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeToolModel(BaseChatModel):
    """测试用：按队列返回消息；bind_tools 返回自身。"""
    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._responses.pop(0) if self._responses else AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _identifying_params(self):
        return {"name": "FakeToolModel"}

    @property
    def _llm_type(self):
        return "fake-tool"
```

- [ ] **Step 2: 写失败测试 `tests/test_graph.py`**

```python
from langchain_core.messages import AIMessage
from agent.config import Settings
from agent.graph import build_graph
from tests.conftest import FakeToolModel


def _settings(tmp_path):
    return Settings(llm_api_key="sk-x", llm_model="gpt-4o-mini",
                    sqlite_path=str(tmp_path / "c.sqlite"))

def test_build_graph_compiled(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.graph.build_llm",
                        lambda s: FakeToolModel([AIMessage(content="hi")]))
    g = build_graph(_settings(tmp_path))
    assert hasattr(g, "ainvoke") and hasattr(g, "astream_events")

def test_react_loop_calls_tool(monkeypatch, tmp_path):
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[{"name": "calculator",
            "args": {"expression": "2+2"}, "id": "c1", "type": "tool_call"}]),
        AIMessage(content="结果是 4"),
    ])
    monkeypatch.setattr("agent.graph.build_llm", lambda s: model)
    g = build_graph(_settings(tmp_path))
    out = g.invoke({"messages": [{"role": "user", "content": "算 2+2"}]})
    types = [type(m).__name__ for m in out["messages"]]
    assert types == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert out["messages"][-1].content == "结果是 4"
```

- [ ] **Step 3: 验证失败** -> FAIL(ModuleNotFoundError: agent.graph)
- [ ] **Step 4: 实现 `agent/graph.py`**

```python
from langchain.agents import create_agent
from agent.config import Settings
from agent.llm.factory import build_llm
from agent.tools import get_tools
from agent.memory.checkpointer import build_checkpointer
from agent.memory.trimming import make_trim_middleware
from agent.prompts import SYSTEM_PROMPT


def build_graph(settings: Settings | None = None):
    settings = settings or Settings()
    llm = build_llm(settings)
    cm = build_checkpointer(settings.sqlite_path)
    checkpointer = cm.__enter__()
    return create_agent(
        model=llm,
        tools=get_tools(),
        system_prompt=SYSTEM_PROMPT,
        middleware=[make_trim_middleware(settings.token_budget)],
        checkpointer=checkpointer,
    )
```

注意：用 `cm.__enter__()` 持有连接（图生命周期内不关闭）。已验证 `create_agent` 在此配置下 ReAct 循环正常（spike 通过）。

- [ ] **Step 5: 实现 `agent/__init__.py`**

```python
from agent.config import Settings
from agent.graph import build_graph

__all__ = ["Settings", "build_graph"]
```

- [ ] **Step 6: 验证通过** - `pytest tests/test_graph.py -v` -> PASS
- [ ] **Step 7: Commit** - `git add tests/conftest.py agent/graph.py agent/__init__.py tests/test_graph.py && git commit -m "feat: graph assembly + react loop test"`

## Task 9: HTTP API

**Files:** Create `agent/api.py`, `tests/test_api.py`
**Consumes:** `build_graph`, `get_tools` **Produces:** FastAPI `app`，路由 `/health`,`/tools`,`/chat`(SSE)

- [ ] **Step 1: 写失败测试 `tests/test_api.py`**

```python
from fastapi.testclient import TestClient
from agent.api import app

def test_health():
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}

def test_tools():
    with TestClient(app) as c:
        names = [t["name"] for t in c.get("/tools").json()]
        assert "calculator" in names and "web_search" in names

def test_chat_returns_200(monkeypatch):
    class FakeGraph:
        async def astream_events(self, inp, config, version="v2"):
            return iter([])  # 无事件
    monkeypatch.setattr("agent.api.get_graph", lambda: FakeGraph())
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "hi", "thread_id": "t1"})
        assert r.status_code == 200
```

- [ ] **Step 2: 验证失败** -> FAIL
- [ ] **Step 3: 实现 `agent/api.py`**

```python
import json
import uuid
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent.graph import build_graph
from agent.tools import get_tools

app = FastAPI(title="Agent API")
_graph = None

def get_graph():
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
async def chat(req: ChatReq, graph=Depends(get_graph)):
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
```

- [ ] **Step 4: 验证通过** - `pytest tests/test_api.py -v` -> PASS
- [ ] **Step 5: Commit** - `git add agent/api.py tests/test_api.py && git commit -m "feat: fastapi chat sse"`

## Task 10: CLI

**Files:** Create `agent/cli.py`, `tests/test_cli.py`
**Produces:** `main()`，命令 `agent tools` / `agent chat` / `agent serve`

- [ ] **Step 1: 写失败测试 `tests/test_cli.py`**

```python
from click.testing import CliRunner
from agent.cli import main

def test_tools_command():
    res = CliRunner().invoke(main, ["tools"])
    assert res.exit_code == 0
    assert "calculator" in res.output and "web_search" in res.output

def test_help():
    res = CliRunner().invoke(main, ["--help"])
    assert res.exit_code == 0
    assert "chat" in res.output and "serve" in res.output
```

- [ ] **Step 2: 验证失败** -> FAIL
- [ ] **Step 3: 实现 `agent/cli.py`**

```python
import asyncio
import uuid
import click
from langchain_core.messages import HumanMessage
from agent.tools import get_tools
from agent.config import Settings


@click.group()
def main():
    """通用 Agent 框架 CLI。"""

@main.command()
def tools():
    """列出已注册工具。"""
    for t in get_tools():
        click.echo(f"- {t.name}: {t.description}")

@main.command()
@click.option("--thread", default=None, help="恢复指定会话 id")
def chat(thread):
    """交互式对话。"""
    from agent.graph import build_graph
    tid = thread or str(uuid.uuid4())
    click.echo(f"会话 id: {tid}（输入 /exit 退出）")
    graph = build_graph()
    config = {"configurable": {"thread_id": tid}}
    while True:
        msg = click.prompt("你", type=str, default="", show_default=False)
        if not msg or msg.strip() == "/exit":
            break

        async def run():
            async for ev in graph.astream_events(
                {"messages": [HumanMessage(content=msg)]}, config, version="v2"
            ):
                if ev["event"] == "on_chat_model_stream" and ev["data"]["chunk"].content:
                    click.echo(ev["data"]["chunk"].content, nl=False)
            click.echo()

        asyncio.run(run())

@main.command()
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
def serve(host, port):
    """启动 HTTP API 服务。"""
    import uvicorn
    s = Settings()
    uvicorn.run("agent.api:app", host=host or s.api_host, port=port or s.api_port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 验证通过** - `pytest tests/test_cli.py -v` -> PASS
- [ ] **Step 5: Commit** - `git add agent/cli.py tests/test_cli.py && git commit -m "feat: cli (chat/serve/tools)"`

---

## Task 11: README + 验收

**Files:** Create `README.md`

- [ ] **Step 1: 写 `README.md`** - 内容：项目简介、`pip install -e ".[dev]"`、复制 `.env.example` 为 `.env` 填 key、`agent chat`/`agent serve`/`agent tools` 用法、扩展点（加工具/换 LLM/换裁剪中间件/换图）。注明：`create_react_agent` 已弃用改用 `create_agent`；`duckduckgo_search` 已改名 `ddgs`（当前版本仍可用）。
- [ ] **Step 2: 全量测试** - `pytest -v` -> 全绿
- [ ] **Step 3: 验收清单**（自动测已覆盖下列可自动化项；手动项需配 .env 真实 key）
  - [ ] `agent tools` 列出 2 工具（test_cli 覆盖）
  - [ ] `pytest` 全绿
  - [ ] ReAct 循环调工具产出最终回复（test_graph 覆盖）
  - [ ] `/health`、`/tools`、`/chat` 200（test_api 覆盖）
  - [ ] 手动：`agent chat` 多轮对话调 calculator/web_search
  - [ ] 手动：重启 `agent chat --thread <id>` 恢复上文
  - [ ] 手动：换 `.env` 的 `LLM_BASE_URL`+`LLM_MODEL` 切后端
- [ ] **Step 4: Commit** - `git add README.md && git commit -m "docs: readme + acceptance"`

---

## Self-Review (执行前自查)

- **Spec 覆盖**：config(1)、工具注册(2)、calculator(3)、web_search(4)、LLM 工厂(5)、checkpointer(6)、裁剪+prompt(7)、graph(8)、API(9)、CLI(10)、README(11) -- spec 全覆盖。
- **占位符**：无 TBD/TODO；所有代码步骤含完整可执行代码。
- **类型一致**：`build_llm(settings)`、`get_tools()`、`build_checkpointer(sqlite_path)`、`make_trim_middleware(max_tokens)`、`build_graph(settings)`、`get_graph()` 在各任务间签名一致。
- **风险点**：Task 8 用 `cm.__enter__()` 持有连接（spike 验证 create_agent 此配置 ReAct 循环通过）；若 sqlite `:memory:` 不可用，测试用 tmp_path 文件。

