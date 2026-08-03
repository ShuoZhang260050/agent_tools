from typing import Any

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langgraph.types import interrupt

REQUEST_APPROVAL = "request_approval"
AUTO_APPROVE = "auto_approve"
FULL_ACCESS = "full_access"

PERMISSION_CHOICES = [
    {"value": REQUEST_APPROVAL, "label": "请求审批"},
    {"value": AUTO_APPROVE, "label": "替我审批"},
    {"value": FULL_ACCESS, "label": "完全访问权限"},
]

DEFAULT_PERMISSION = REQUEST_APPROVAL

SENSITIVE_TOOLS = {
    "run_command",
    "run_python",
    "write_file",
    "edit_file",
    "download_file",
    "http_request",
    "browser",
    "save_memory",
}


def is_sensitive(name: str) -> bool:
    return name in SENSITIVE_TOOLS


def permission_prompt_section(permission: str) -> str:
    sensitive = "、".join(sorted(SENSITIVE_TOOLS))
    if permission == REQUEST_APPROVAL:
        return (
            "\n\n<permission>\n"
            f"当前权限模式：请求审批。在执行以下敏感工具（{sensitive}）前，"
            "系统会暂停并请求用户确认；只有收到确认后才会真正执行。"
            "只读类工具（calculator、web_search、read_file 等）可直接执行。"
            "调用敏感工具前请先用一句话说明意图。\n"
            "</permission>"
        )
    if permission == AUTO_APPROVE:
        return (
            "\n\n<permission>\n"
            "当前权限模式：替我审批。敏感工具将由系统自动审批并执行，"
            "执行后请简要告知用户你做了什么。\n"
            "</permission>"
        )
    return (
        "\n\n<permission>\n"
        "当前权限模式：完全访问权限。所有工具可直接执行，无需额外审批或说明。\n"
        "</permission>"
    )


@wrap_tool_call(name="PermissionGateMiddleware")
def permission_gate(request, handler):
    tool_call = request.tool_call or {}
    name = tool_call.get("name", "")
    call_id = tool_call.get("id", "")

    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    permission = configurable.get("permission", DEFAULT_PERMISSION)

    if is_sensitive(name) and permission == REQUEST_APPROVAL:
        decision = interrupt({
            "tool": name,
            "args": tool_call.get("args", {}),
            "tool_call_id": call_id,
        })
        if decision != "approved":
            return ToolMessage(
                content=f"用户已取消此操作（{name}），未执行。",
                tool_call_id=call_id,
                name=name,
            )

    return handler(request)


def find_pending_approval(graph, config) -> dict | None:
    try:
        state = graph.get_state(config)
    except Exception:
        return None
    for task in getattr(state, "tasks", []) or []:
        for intr in getattr(task, "interrupts", []) or []:
            val = getattr(intr, "value", None)
            if isinstance(val, dict) and val.get("tool"):
                return val
    return None
