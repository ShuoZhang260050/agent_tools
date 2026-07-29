# Agent Tools - 可扩展 LLM Agent 框架

一个基于 [LangChain](https://github.com/langchain-ai/langchain) `create_agent` 的通用、可扩展 LLM Agent 框架。提供 Python 库、命令行（CLI）和 HTTP API 三种入口，内置 SQLite 会话持久化、滑动窗口上下文裁剪，以及可插拔的工具/LLM/记忆机制。

## 特性

- **ReAct 工具循环**：基于 `langchain.agents.create_agent`，LLM 自主决策调用工具 -> 观察结果 -> 再决策
- **用户认证**：JWT 登录/注册，用户隔离的记忆与会话
- **RAG 知识库**：文档上传（txt/md/pdf）-> 分块 -> 向量化 -> 语义检索，按 user 隔离
- **可观测性**：TracingCallbackHandler 记录每次 LLM/工具调用的耗时与 token 用量，前端调试面板可视化
- **可插拔 LLM**：一条 OpenAI 兼容路径覆盖官方 OpenAI、智谱 GLM、通义千问、DeepSeek、本地 Ollama 等
- **三种入口**：`import` 库 / `agent` CLI / FastAPI HTTP API（SSE 流式）
- **SQLite 持久记忆**：按 `thread_id` 隔离，重启不丢上下文
- **上下文工程栈**：结构化系统提示词（XML+Markdown）+ 用户记忆注入 + 摘要压缩 + 状态栏 + 任务追踪 + 预算控制 + 提示注入防御
- **装饰器工具注册**：加工具只需 `@register @tool`，无需改中央清单
- **完整测试**：76 个测试覆盖工具、ReAct 循环、认证、记忆隔离、RAG、追踪、API/CLI

## 安装

需要 Python 3.12+。

```bash
# 克隆后进入项目目录
python -m venv .venv
# Windows PowerShell
$env:Path = ".\.venv\Scripts;$env:Path"
# Linux/macOS
# source .venv/bin/activate

pip install -e ".[dev]"
```

## 配置

复制 `.env.example` 为 `.env` 并填入你的 LLM 配置：

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...
# 留空=官方 OpenAI 端点；国产/本地模型填兼容端点：
# 智谱 GLM:    https://open.bigmodel.cn/api/paas/v4
# 通义千问:    https://dashscope.aliyuncs.com/compatible-mode/v1
# DeepSeek:    https://api.deepseek.com
# 本地 Ollama: http://localhost:11434/v1
LLM_BASE_URL=
LLM_TEMPERATURE=0.7
TOKEN_BUDGET=6000
SQLITE_PATH=checkpoints.sqlite
API_HOST=0.0.0.0
API_PORT=8000
```

## 使用

### CLI

```bash
agent tools              # 列出已注册工具
agent chat               # 交互式对话（生成新会话 id，流式输出）
agent chat --thread <id> # 恢复指定会话（沿用历史上下文）
agent serve              # 启动 HTTP API 服务（默认 0.0.0.0:8000）
agent serve --port 9000  # 指定端口
```

### HTTP API

```bash
agent serve
# 另一个终端：
curl http://localhost:8000/health          # {"status":"ok"}
curl http://localhost:8000/tools           # [{"name":"calculator",...},...]

# 流式对话（SSE）
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我算 123 * 456", "thread_id": "demo"}'
# data: {"token": ...}   （逐 token 流式）
# data: {"thread_id": "demo"}
```

### 作为库

```python
from agent import build_graph, Settings

graph = build_graph()  # 读 .env
out = graph.invoke(
    {"messages": [{"role": "user", "content": "算 2+2"}]},
    config={"configurable": {"thread_id": "my-session"}},
)
print(out["messages"][-1].content)
```

## 架构

```
前端：Library / CLI (click) / API (FastAPI SSE)
        │  都只调用 graph
        ▼
核心：create_agent(model, tools, [dynamic_prompt, todo, limit, summarization, trim, state_bar], checkpointer)
        │
   ┌────┴────────┬─────────────┬──────────────────┐
   ▼             ▼             ▼                  ▼
 LLM 工厂       工具注册表      记忆层              配置
 build_llm     get_tools()     SQLite checkpointer  Settings
 (OpenAI兼容)  (@register @tool) + user_memory      (.env)
                               trim + state_bar
```

- `agent/config.py` — pydantic-settings 读取 `.env`，单一配置源
- `agent/llm/factory.py` — 按 `LLM_PROVIDER` 分派构建 LLM
- `agent/tools/` — `registry.py` 注册表 + `calculator.py` / `web_search.py` 示例工具
- `agent/memory/` — `checkpointer.py`（SQLite）+ `trimming.py`（滑动窗口中间件）
- `agent/prompts.py` — system prompt 常量
- `agent/graph.py` — `build_graph()` 装配
- `agent/cli.py` / `agent/api.py` — 前端

## 扩展点

| 想做什么 | 改哪里 |
|----------|--------|
| 新增工具 | `agent/tools/` 下加模块，`@register @tool` 装饰，在 `tools/__init__.py` import |
| 换 LLM 后端 | `agent/llm/factory.py` 加一个 `llm_provider` 分支 |
| 摘要压缩替代滑动窗口 | 替换 `agent/memory/trimming.py` 的 `make_trim_middleware`（或用库内置 `SummarizationMiddleware`） |
| 自定义状态栏 | 改 `agent/memory/state_bar.py` 的 `_build_state_bar` |
| 调整预算/摘要阈值 | 改 `agent/config.py` 的 `model_call_limit`/`summary_trigger_messages`/`summary_keep_messages` |
| 跨会话用户记忆 | 已内置 `agent/memory/user_memory.py` + `save_memory` 工具，按 `user_id` 隔离 |
| 自定义图/多智能体 | 把 `graph.build_graph` 内部换成手写 `StateGraph`，上层 CLI/API 不变 |
| 动态 system prompt | 改用 `langchain.agents.middleware` 的 `dynamic_prompt` 中间件（已在 `graph.py` 中用于注入用户记忆） |

## 测试

```bash
pytest -v          # 全部测试
```

测试不打真实 API：工具单测 mock 外部调用；ReAct 循环测试用 `FakeToolModel`（支持 `bind_tools` 的测试替身）验证「调工具 -> 观察 -> 再答」全程；API/CLI 测试用 mock 图。

## 备注

- 使用 `langchain.agents.create_agent`（`create_react_agent` 已被官方弃用，`create_agent` 是推荐替代，其 middleware 系统更契合本框架的可扩展目标）。
- `duckduckgo_search` 包已更名为 `ddgs`；当前安装的 `duckduckgo_search` 8.x 仍可用（运行时会打印一条重命名提示）。若后续升级，将 `web_search.py` 的 `from duckduckgo_search import DDGS` 改为 `from ddgs import DDGS` 即可。

## 文档

- 设计文档：`docs/specs/2026-07-28-agent-framework-design.md`
- 实现计划：`docs/plans/2026-07-28-agent-framework.md`
