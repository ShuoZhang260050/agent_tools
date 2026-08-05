# AGENTS.md

AI Agent 在本项目的操作指南。进入项目后请先通读本文件。

## 项目概述

基于 LangChain `create_agent` 的可扩展 LLM Agent 框架。提供 Python 库、CLI、HTTP API 三种入口，内置 Shadow Workspace 沙箱、权限审批、RAG 知识库、SQLite 会话持久化。

## 技术栈

- Python 3.12+
- LangChain 1.3+ / LangGraph（`create_agent` + middleware 系统）
- FastAPI + SSE 流式
- SQLite（会话持久化、用户记忆、快照）
- pydantic-settings（`.env` 配置）
- pytest（测试）
- 前端：原生 JS + SSE（无框架）

## 常用命令

```bash
# 安装
pip install -e ".[dev]"

# 测试（必须全绿才能提交）
python -m pytest tests/ -q

# 启动 API 服务
agent serve

# CLI 对话
agent chat

# 前端 JS 语法检查（修改 index.html 后运行）
# 从 index.html 提取 <script> 内容到临时文件后：
node --check <file>
```

## 项目结构

```
agent/
  config.py          # Settings：所有配置项，读 .env
  prompts.py         # 系统提示词常量（含 shadow_workspace 指令）
  graph.py           # build_graph() 装配中间件链
  permissions.py     # 权限模型：请求审批/替我审批/完全访问
  auth.py            # JWT 认证
  api.py             # FastAPI HTTP API + SSE 流式
  cli.py             # click CLI
  llm/factory.py     # LLM 工厂，按 provider 分派
  tools/             # 工具注册表 + 工具实现
    registry.py      # @register 装饰器
    file_ops.py      # write_file / edit_file
    read_file.py     # read_file
    list_files.py    # list_files
    search_files.py  # search_files
    run_command.py   # run_command
    run_python.py    # run_python
    ...              # calculator/web_search/weather/http_request 等
  memory/
    user_memory.py   # 用户记忆、工作空间、会话管理
    checkpointer.py  # SQLite checkpointer
    trimming.py      # 滑动窗口上下文裁剪中间件
    state_bar.py     # 状态栏中间件
    tracing.py       # TracingCallbackHandler
  sandbox/
    shadow.py        # ShadowManager: 过滤拷贝/diff/sync/verify + 注册表
    snapshot.py      # file_snapshots 表: 快照备份/恢复/回滚
  middleware/
    shadow_gate.py   # turn 首次工具调用时创建 shadow
  static/
    index.html       # 前端单页（JS + CSS 内联）
tests/
  conftest.py        # FakeToolModel 测试替身
  test_*.py          # 216 个测试
```

## 代码约定

- 工具注册：`@register` 装饰器，无需改中央清单
- 路径处理：统一用 `pathlib.Path`，不用 `os.path` 拼接
- 工具内 workspace 解析：`get_active_workspace(uid, tid) or get_workspace(uid)` 一行，shadow 优先
- 错误消息：中文，面向用户可读
- Docstring：函数和类主动添加 docstring（中文，一句话说明用途）
- 注释：不主动添加行内注释，除非逻辑复杂或用户要求
- 系统提示词：XML 标签分区（`<role>`/`<rules>`/`<shadow_workspace>` 等）
- 提交消息：`feat:` / `fix:` / `refactor:` / `docs:` 前缀

## Shadow Workspace 沙箱

所有文件操作和命令执行在 shadow 副本上进行，不直接修改真实工作空间。

**关键机制**：
- `shadow_gate` 中间件在 turn 首次工具调用时懒创建 shadow 副本（过滤 `.git`/`node_modules`/`.venv` 等，尊重 `.gitignore`，上限 200MB）
- 所有 6 个工具（file_ops/read_file/list_files/search_files/run_command/run_python）通过 `get_active_workspace()` 重定向到 shadow
- Turn 结束时 `done` 事件携带 `pending_sync` + diff
- 前端 sync 面板：验证（pytest/ruff）-> 同步（先快照再 apply）或 拒绝（从快照恢复）
- 系统提示词要求 agent 编写代码后必须运行测试，测试未通过不能宣称完成

**修改 shadow 配置**：改 `.env` 中 `SHADOW_MAX_BYTES` / `SHADOW_SKIP_DIRS` / `SHADOW_VERIFY_TIMEOUT`

## 测试约定

- 不调真实 LLM：用 `tests/conftest.py` 的 `FakeToolModel`（按队列返回消息，支持 `bind_tools`）
- 隔离：`tmp_path` + `monkeypatch`，每个测试独立工作空间和数据库
- API 测试：`TestClient(app)` + `app.dependency_overrides[get_current_user]` 覆盖认证
- 工具测试：直接调用 `tool._run()`，不走 graph 中间件
- 中间件测试：走 `graph.invoke()`，验证 shadow 创建、工具重定向
- 新增功能必须附带测试

## 权限模型

- `request_approval`（默认）：敏感工具执行前 `interrupt()` 暂停，等用户确认
- `auto_approve`：系统自动审批敏感工具
- `full_access`：无限制
- 敏感工具集合：`run_command` / `run_python` / `write_file` / `edit_file` / `download_file` / `http_request` / `browser` / `save_memory`

## 配置项

所有配置在 `.env` 中（参考 `.env.example`）。关键项：

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MODEL` | `gpt-4o-mini` | 模型名 |
| `SQLITE_PATH` | `checkpoints.sqlite` | SQLite 数据库路径 |
| `SHADOW_MAX_BYTES` | `209715200` | shadow 副本大小上限（200MB） |
| `SHADOW_SKIP_DIRS` | `.git,node_modules,...` | shadow 跳过的目录 |
| `SHADOW_VERIFY_TIMEOUT` | `120` | 验证命令超时秒数 |
| `TOKEN_BUDGET` | `500000` | 上下文 token 上限 |
| `MODEL_CALL_LIMIT` | `25` | 单轮模型调用上限 |

## 开发工作流程

接到需求后按以下步骤执行：

### 1. 拆任务

用 `todowrite` 将需求拆分为可执行的子任务（3+ 步骤时必须拆）。每个任务标注 priority 和 status。

### 2. 逐步实现

按任务列表逐步实现。每完成一个任务，立即更新 `todowrite` 标记为 `completed`，再开始下一个。同一时间只有一个 `in_progress`。

### 3. 测试

实现完成后运行测试：

```bash
python -m pytest tests/ -q
```

- 必须全绿才能提交
- 新增功能必须附带测试
- 修改已有功能后运行全量回归确保无破坏

### 4. 更新文档

若改动涉及以下任一变更，必须同步更新对应文档：

| 变更类型 | 更新文件 |
|----------|----------|
| Agent 行为/指令变化 | `agent/prompts.py`（系统提示词） |
| 新增/修改 API 端点 | `README.md` + `agent/prompts.py`（如影响 agent 行为） |
| 新增/修改配置项 | `.env.example` + `README.md` |
| 新增模块/目录 | `README.md` 项目结构 |
| 特性增减 | `README.md` 特性列表 + 测试数 |

### 5. 提交

测试通过 + 文档更新后，提交相关文件：

```bash
git add <相关文件>
git commit -m "feat: 简述"   # 或 fix:/refactor:/docs:
```

- 不要提交 `.env`、`checkpoints.sqlite`、`*.log` 等运行时文件
- 不要自动推送，除非用户明确要求

## 常见任务指引

| 任务 | 修改位置 |
|------|----------|
| 新增工具 | `agent/tools/` 下新建模块，`@register` 装饰，在 `tools/__init__.py` import |
| 换 LLM 后端 | `agent/llm/factory.py` 加 `llm_provider` 分支 |
| 改 shadow 大小/跳过目录 | `.env` 中 `SHADOW_MAX_BYTES` / `SHADOW_SKIP_DIRS` |
| 改默认验证命令 | `agent/static/index.html` 中 `sync-verify-cmd` 的 `value` |
| 调整上下文预算 | `.env` 中 `TOKEN_BUDGET` / `MODEL_CALL_LIMIT` |
| 跨会话用户记忆 | 已内置 `agent/memory/user_memory.py` + `save_memory` 工具 |
