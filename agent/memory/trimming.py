from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, trim_messages


def _repair_tool_pairing(messages):
    """裁剪后修复 tool 消息配对，避免发给 LLM 的消息序列非法导致 400。

    - 移除孤立的 ToolMessage（对应的 AIMessage(tool_calls) 已被裁掉）
    - 移除末尾有 tool_calls 但无后续 ToolMessage 的 AIMessage
    """
    valid_ids = {
        tc["id"]
        for m in messages
        for tc in (getattr(m, "tool_calls", None) or [])
        if tc.get("id")
    }
    repaired = [
        m
        for m in messages
        if not (isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None) not in valid_ids)
    ]
    while repaired and getattr(repaired[-1], "tool_calls", None):
        repaired.pop()
    return repaired


def _ensure_user_start(trimmed, original):
    """确保消息序列以 user/system 开头。

    glm-5.2 等 OpenAI 兼容端点要求 messages 不能以 assistant 开头，否则返回
    400 InvalidParameter。trim(strategy='last') 可能把开头的 user 裁掉，只剩
    [assistant(tool_calls), tool]，导致 400。此处向前从 original 补最近一个 user。
    """
    if not trimmed:
        return trimmed
    if isinstance(trimmed[0], (HumanMessage, SystemMessage)):
        return trimmed
    start_idx = None
    for i, m in enumerate(original):
        if m is trimmed[0]:
            start_idx = i
            break
    if start_idx is not None:
        for j in range(start_idx - 1, -1, -1):
            if isinstance(original[j], HumanMessage):
                return [original[j]] + trimmed
    return trimmed


def make_trim_middleware(max_tokens: int):
    """创建滑动窗口上下文裁剪中间件。"""
    @wrap_model_call(name="TrimMessagesMiddleware")
    def trim(request, handler):
        """裁剪中间件实现，按 token 预算滑动窗口。"""
        trimmed = trim_messages(
            request.messages,
            strategy="last",
            token_counter="approximate",
            max_tokens=max_tokens,
        )
        trimmed = _repair_tool_pairing(trimmed)
        trimmed = _ensure_user_start(trimmed, request.messages)
        if not trimmed:
            trimmed = _repair_tool_pairing(request.messages)
        return handler(request.override(messages=trimmed))

    return trim
