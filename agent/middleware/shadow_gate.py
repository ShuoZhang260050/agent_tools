from langchain.agents.middleware import wrap_tool_call

SANDBOX_TOOLS = {
    "write_file", "edit_file", "read_file", "list_files",
    "search_files", "run_command", "run_python",
}


@wrap_tool_call(name="ShadowGateMiddleware")
def shadow_gate(request, handler):
    """Shadow 工作空间中间件，turn 首次沙箱工具调用时创建 shadow 副本。"""
    tool_call = getattr(request, "tool_call", None) or {}
    name = tool_call.get("name", "")
    if name not in SANDBOX_TOOLS:
        return handler(request)

    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    user_id = configurable.get("user_id")
    thread_id = configurable.get("thread_id")

    if user_id and thread_id:
        from agent.sandbox.shadow import create_shadow_if_needed
        try:
            create_shadow_if_needed(user_id, thread_id)
        except Exception as e:
            raise RuntimeError(f"Shadow 工作空间创建失败：{e}") from e

    return handler(request)
