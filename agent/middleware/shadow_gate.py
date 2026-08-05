from langchain.agents.middleware import wrap_tool_call


@wrap_tool_call(name="ShadowGateMiddleware")
def shadow_gate(request, handler):
    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    user_id = configurable.get("user_id")
    thread_id = configurable.get("thread_id")

    if user_id and thread_id:
        from agent.sandbox.shadow import create_shadow_if_needed
        try:
            create_shadow_if_needed(user_id, thread_id)
        except Exception:
            pass

    return handler(request)
