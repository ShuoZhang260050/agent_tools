# 可扩展 Agent 框架设计文档

- **日期**: 2026-07-28
- **状态**: 设计已确认，待实现
- **技术栈**: Python 3.12 / LangGraph 1.2 / LangChain 1.3 / FastAPI / SQLite

## 1. 目标与范围

构建一个**通用、可扩展的 LLM Agent 框架**作为可运行骨架。起步即提供：

- 基于 LangGraph ReAct 模式的 agent 核心（LLM 决策 → 调工具 → 观察 → 再决策的循环）
- 可插拔的 LLM 后端（OpenAI 兼容，一条路径覆盖官方 OpenAI、国产模型、本地模型）
- 三种交互入口：可导入的 Python 库 + 命令行 REPL + HTTP API（流式）
- SQLite 持久化的会话记忆
- 清晰的工具注册机制，内置 2 个示例工具演示扩展方式
- 完整测试覆盖

**非目标（YAGNI，起步不做）**：多智能体协作、人工介入(human-in-the-loop)、跨会话长期记忆(`Store`)、代码沙箱执行、RAG 检索增强。这些在架构中预留扩展点，但不实现。

## 2. 关键决策

| 维度 | 决策 | 理由 |
|------|------|------|
| Agent 核心 | LangGraph 预置 `create_react_agent`（ReAct） | 最少代码跑通，工具/LLM/记忆可插拔；预留换图接缝 |
| LLM 接入 | 可配置多后端，OpenAI 兼容路径为主 | 一条路径覆盖官方/国产/本地模型 |
| 交互方式 | 库 + CLI + HTTP API 全要 | 核心是 graph，前后端都是薄封装 |
| 内置工具 | Web 搜索 + 计算器 | 演示机制，无安全风险 |
| 会话记忆 | SQLite 持久化（langgraph-checkpoint-sqlite） | 重启不丢，按 thread_id 隔离 |
| 上下文裁剪 | 滑动窗口（tiktoken 计 token，保留 system + 最近 N） | 简单、确定、零延迟；接口预留可换摘要压缩 |

## 3. 架构总览

```
┌─────────────────────────────────────────────────────────┐
│  前端层 (薄封装，都只调 graph)                            │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐   │
│  │  Library │   │  CLI (click) │   │ API (FastAPI)  │   │
│  │ agent.run│   │ agent chat   │   │ POST /chat SSE │   │
│  └────┬─────┘   └──────┬───────┘   └───────┬────────┘   │
└───────┼────────────────┼───────────────────┼────────────┘
        │                │                   │
        └────────────────┼───────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  核心层  graph = build_graph()                           │
│  create_react_agent(llm, tools, checkpointer,            │
│                     state_modifier[prompt+裁剪])          │
└───┬──────────┬──────────────┬───────────────────────────┘
    ▼          ▼              ▼
┌────────┐ ┌────────┐ ┌──────────────────┐
│ LLM 工厂│ │ 工具   │ │ 记忆层           │
│factory │ │registry│ │ checkpointer(SQLite)│
│        │ │        │ │ trimming(滑动窗口) │
└───┬────┘ └───┬────┘ └────────┬─────────┘
    │          │               │
    ▼          ▼               ▼
 config (pydantic-settings 读取 .env)
```

依赖方向：前端 → 核心 → (LLM 工厂 / 工具注册表 / 记忆层) → config。每层单向依赖，可独立测试。

## 4. 项目结构

```
agent_tools/
├── .env                      # 配置（见第 6 节）
├── pyproject.toml            # 依赖 + console_scripts
├── .gitignore
├── README.md
├── docs/specs/               # 本文档所在
├── agent/
│   ├── __init__.py           # 暴露 build_graph / run 等顶层 API
│   ├── config.py             # pydantic-settings Settings
│   ├── llm/
│   │   ├── __init__.py
│   │   └── factory.py        # build_llm(settings) -> BaseChatModel
│   ├── tools/
│   │   ├── __init__.py       # 显式 import 各工具模块触发注册
│   │   ├── registry.py       # @register 装饰器 + get_tools()
│   │   ├── web_search.py     # DuckDuckGo 搜索
│   │   └── calculator.py     # 安全表达式计算
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── checkpointer.py   # build_checkpointer(settings) -> BaseCheckpointSaver
│   │   └── trimming.py       # make_state_modifier(budget) 滑动窗口裁剪
│   ├── prompts.py            # system_prompt(state) -> str 函数
│   ├── graph.py              # build_graph(settings) -> CompiledGraph
│   ├── cli.py                # click 命令: chat / serve / tools
│   └── api.py                # FastAPI app + 路由
└── tests/
    ├── conftest.py
    ├── test_tools.py
    ├── test_graph.py
    └── test_api.py
```

## 5. 组件详细设计

### 5.1 config.py — 配置层

使用 `pydantic-settings` 的 `BaseSettings`，从 `.env` 与环境变量读取。所有组件都依赖这个单一配置源。

```python
class Settings(BaseSettings):
    # LLM
    llm_provider: str = "openai"          # 预留分派键，起步仅实现 openai 兼容路径
    llm_model: str = "gpt-4o-mini"
    llm_api_key: SecretStr
    llm_base_url: str | None = None       # 国产/本地兼容端点
    llm_temperature: float = 0.7
    # 上下文
    token_budget: int = 6000              # 滑动窗口保留的最近消息 token 上限
    # 记忆
    sqlite_path: str = "checkpoints.sqlite"
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_config = SettingsConfigDict(env_file=".env", env_prefix="")
```

### 5.2 llm/factory.py — LLM 工厂

按 `llm_provider` 分派，起步实现 `openai` 分支（一条路径覆盖官方 OpenAI / 智谱GLM / 通义 / DeepSeek / 本地 Ollama(localhost:11434/v1)——只要暴露 OpenAI 兼容端点即可）。

```python
def build_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,        # None 时用官方端点
            temperature=settings.llm_temperature,
        )
    raise ValueError(f"未知 provider: {settings.llm_provider}")
```

**新增 provider**：在 `factory.py` 加一个 `elif` 分支，引入对应的 `ChatXxx` 即可。文档(README)将给出示例。

### 5.3 tools/ — 工具注册机制

装饰器注册表。加工具 = 新建模块 + `@register` `@tool` 装饰，无需改中央清单。

```python
# tools/registry.py
from typing import Callable
_REGISTRY: list = []

def register(tool):
    """注册一个工具到全局表。"""
    _REGISTRY.append(tool)
    return tool

def get_tools():
    """返回已注册工具的副本。"""
    return list(_REGISTRY)
```

```python
# tools/web_search.py
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from .registry import register

@register
@tool
def web_search(query: str) -> str:
    """搜索网络获取最新信息。输入搜索关键词。"""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    return "\n".join(f"- {r['title']}: {r['body']}" for r in results) or "无结果"
```

`tools/__init__.py` 显式 `from . import web_search, calculator` 以触发注册（**不**做魔法自动扫描未知模块，保持可预测）。`get_tools()` 返回的列表喂给 `create_react_agent`。

**calculator.py**：用 `ast` 安全解析算术表达式（仅允许数字与运算符，拒绝名字/调用），避免 `eval` 安全风险。

### 5.4 memory/ — 记忆与上下文处理

上下文处理分三层：

**第 1 层 — 会话内历史（thread-scoped memory，内置）**
- 每轮对话 = 一个 `thread_id`。`create_react_agent` 的 state 中 `messages` 列表由 SQLite checkpointer 按 `thread_id` 持久化。
- **CLI**：`agent chat` 启动生成 uuid `thread_id`；`--thread <id>` 可恢复指定会话。
- **API**：客户端请求带 `thread_id`，服务端用 `{"configurable": {"thread_id": ...}}` 调 graph，自动加载该线程完整历史。

**第 2 层 — 上下文窗口管理（内置，滑动窗口）**
- `create_react_agent` 的 `state_modifier` 钩子，在每次调 LLM 前裁剪 messages。
- 用 `langchain_core.messages.trim_messages(strategy="last", token_counter=tiktoken, max_tokens=settings.token_budget)`，保留 system prompt + 最近 N 个 token 的消息，旧消息丢弃。
- 接口包成 `make_state_modifier(budget, system_prompt)`：该修饰函数先置 system message 于首，再裁剪。只向 `create_react_agent` 传这一个 `state_modifier`，**不**再单独传 `prompt=`，规避二者同传在不同 langgraph-prebuilt 版本间的歧义。将来换"摘要压缩"只改这一个函数。

**第 3 层 — 跨会话长期记忆（不实现，预留）**
- LangGraph `Store` API（按用户命名空间跨 thread 共享事实/偏好）。架构留注入点，文档标注为扩展点。

```python
# memory/checkpointer.py
def build_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    return SqliteSaver.from_conn_string(settings.sqlite_path)  # 上下文管理连接

# memory/trimming.py
def make_state_modifier(budget: int, system_prompt):
    def modifier(state):
        messages = [SystemMessage(content=system_prompt(state))] + state["messages"]
        return trim_messages(
            messages,
            strategy="last",
            token_counter=tiktoken,
            max_tokens=budget,
            include_system=True,
        )
    return modifier
```

### 5.5 prompts.py — System Prompt

函数式，接收 state、返回 system message 字符串。起步为静态提示，但用函数形式包好，便于将来注入动态上下文（当前日期、用户信息、检索知识）。

```python
def system_prompt(state) -> str:
    return "你是一个乐于助人的助手。可调用工具辅助回答。"
```

### 5.6 graph.py — 核心装配

```python
def build_graph(settings: Settings | None = None):
    settings = settings or Settings()
    llm = build_llm(settings)
    tools = get_tools()
    checkpointer = build_checkpointer(settings)
    return create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        state_modifier=make_state_modifier(settings.token_budget, system_prompt),
    )
```

### 5.7 cli.py — 命令行（click）

- `agent chat [--thread ID] [--model NAME]`：交互式 REPL，读入用户输入，`astream_events` 流式打印 token，`/exit` 退出。无 `--thread` 则生成 uuid。
- `agent serve [--host 0.0.0.0] [--port 8000]`：起 uvicorn 跑 API。
- `agent tools`：列出已注册工具名与描述（便于调试/扩展）。

### 5.8 api.py — HTTP API（FastAPI）

- `POST /chat`：body `{message: str, thread_id?: str}`，走 **SSE 流式**（`astream_events` 逐 token 推送）。无 `thread_id` 则生成。
- `GET /health`：返回 `{"status": "ok"}`。
- `GET /tools`：返回已注册工具列表。
- 开发期启用 CORS（允许本地前端联调）。

## 6. .env 配置项

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...
LLM_BASE_URL=                # 留空=官方；国产/本地填兼容端点
LLM_TEMPERATURE=0.7
TOKEN_BUDGET=6000
SQLITE_PATH=checkpoints.sqlite
API_HOST=0.0.0.0
API_PORT=8000
```

## 7. 数据流

**一次对话请求**（以 API 为例）：
1. 客户端 `POST /chat {message, thread_id}`。
2. 服务端用 `thread_id` 构造 `config={"configurable": {"thread_id": ...}}`。
3. `graph.astream_events({"messages": [HumanMessage(message)]}, config=config)`。
4. checkpointer 自动加载该线程历史 → 与新消息合并 → `state_modifier` 滑动窗口裁剪 → 送入 LLM。
5. LLM 若决策调工具，LangGraph 执行工具节点，结果作为 `ToolMessage` 回灌，LLM 再决策。
6. 每个生成 token 经 SSE 推回客户端；最终 state（含新消息）写回 SQLite。

## 8. 错误处理

- **LLM 调用失败**（网络/鉴权/限流）：捕获并向上抛 `RuntimeError`，CLI 打印友好错误，API 返回 `502` + 错误信息；不吞异常。
- **工具执行异常**：在工具内部捕获，返回形如 `"[工具错误] xxx"` 的字符串作为 `ToolMessage`，让 LLM 据此回应用户（ReAct 惯例），而非中断循环。
- **裁剪兜底**：`trim_messages` 在预算过小时至少保留 system + 最新一条用户消息。
- **SQLite 不可写**：启动期 `build_checkpointer` 建连失败即快速报错，不静默降级。

## 9. 测试策略

| 层 | 文件 | 方法 |
|----|------|------|
| 工具 | test_tools.py | mock `duckduckgo`；calculator 正常/非法表达式 |
| 核心 | test_graph.py | 用 `FakeChatModel`（langchain-core 自带）构造"需要调工具"的响应，断言 ReAct 循环走到工具节点并产出最终回复；全程不碰真实 API |
| API | test_api.py | FastAPI `TestClient` 测 `/health`、`/tools`、`/chat`（mock graph） |
| 配置 | (并入上述) | 用 `.env`/环境变量覆盖验证 Settings 解析 |

## 10. 依赖清单

**新增**：
- `fastapi` — API 框架
- `uvicorn[standard]` — ASGI 服务器
- `langgraph-checkpoint-sqlite` — SQLite checkpointer
- (dev) `pytest`、`pytest-asyncio`

**已就位**：langchain、langchain-core、langchain-openai、langgraph、langgraph-prebuilt、click、pydantic-settings、python-dotenv、SQLAlchemy、duckduckgo_search、tiktoken、httpx。

`pyproject.toml` 定义 console_scripts：
```toml
[project.scripts]
agent = "agent.cli:main"
```

## 11. 扩展点（为未来预留）

| 想做什么 | 改哪里 |
|----------|--------|
| 新增工具 | `tools/` 下加模块 + `@register @tool`，在 `tools/__init__.py` import |
| 换 LLM 后端 | `llm/factory.py` 加 provider 分支 |
| 摘要压缩替代滑动窗口 | 替换 `memory/trimming.py` 的 `make_state_modifier` |
| 跨会话长期记忆 | 接 LangGraph `Store`，在 prompt 注入；架构已留口 |
| 多智能体/自定义图 | 把 `graph.build_graph` 内部换成手写 `StateGraph`，上层 CLI/API 不变 |
| 动态 system prompt | 在 `prompts.system_prompt` 中读 state 注入日期/用户/知识 |
| RAG 检索 | 作为工具注册（`@register @tool def retrieve(...)`）即可融入 ReAct |

## 12. 验收标准

1. `agent chat` 能多轮对话，调用 web_search/calculator 工具，流式输出。
2. 重启后 `agent chat --thread <旧id>` 能恢复上文。
3. `agent serve` 后 `POST /chat` SSE 流式返回。
4. `agent tools` 列出 2 个内置工具。
5. 换 `.env` 的 `LLM_BASE_URL`+`LLM_MODEL` 即可切到国产/本地模型。
6. `pytest` 全绿。
