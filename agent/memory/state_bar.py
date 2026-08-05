from datetime import datetime
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import SystemMessage


def _count_tool_calls(messages) -> int:
    """统计消息序列中工具调用总次数。"""
    count = 0
    for m in messages:
        tcs = getattr(m, "tool_calls", None) or []
        count += len(tcs)
    return count


def _build_state_bar(messages) -> str:
    """构建状态栏文本（含模型/权限/上下文信息）。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    tool_count = _count_tool_calls(messages)
    msg_count = len(messages)
    return (
        f"<state_bar>\n"
        f"当前时间: {now}\n"
        f"对话消息数: {msg_count}\n"
        f"本轮工具调用: {tool_count} 次\n"
        f"</state_bar>"
    )


def make_state_bar_middleware():
    """状态栏中间件：在消息末尾注入动态元信息（时间、工具调用计数）。

    注入位置在消息末尾，不破坏静态前缀，对 KV Cache 友好。
    """
    @wrap_model_call(name="StateBarMiddleware")
    def state_bar(request, handler):
        """状态栏中间件，注入状态栏到系统消息。"""
        msgs = list(request.messages)
        info = _build_state_bar(msgs)
        msgs.append(SystemMessage(content=info))
        return handler(request.override(messages=msgs))
    return state_bar
