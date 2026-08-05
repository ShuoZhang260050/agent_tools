from typing import Any

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langgraph.types import interrupt

from agent.middleware.shadow_gate import SANDBOX_TOOLS

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
    """判断工具是否为敏感工具。"""
    return name in SENSITIVE_TOOLS


def permission_prompt_section(permission: str) -> str:
    """生成权限说明的提示词片段。"""
    sandbox = "、".join(sorted(SENSITIVE_TOOLS & SANDBOX_TOOLS))
    non_sandbox = "、".join(sorted(SENSITIVE_TOOLS - SANDBOX_TOOLS))
    if permission == REQUEST_APPROVAL:
        return (
            "\n\n<permission>\n"
            f"当前权限模式：请求审批。沙箱工具（{sandbox}）在 Shadow 副本中执行，"
            "无需逐条审批，最终变更通过同步面板统一确认。"
            f"以下非沙箱敏感工具（{non_sandbox}）执行前会暂停并请求用户确认。"
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
    """权限审批中间件，敏感工具执行前暂停等待确认。"""
    tool_call = request.tool_call or {}
    name = tool_call.get("name", "")
    call_id = tool_call.get("id", "")

    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    permission = configurable.get("permission", DEFAULT_PERMISSION)

    if is_sensitive(name) and permission == REQUEST_APPROVAL and name not in SANDBOX_TOOLS:
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


def find_all_pending_approvals(graph, config) -> list[dict]:
    """查找所有待审批的中断列表（含 interrupt_id）。"""
    try:
        state = graph.get_state(config)
    except Exception:
        return []
    results = []
    for task in getattr(state, "tasks", []) or []:
        for intr in getattr(task, "interrupts", []) or []:
            val = getattr(intr, "value", None)
            if isinstance(val, dict) and val.get("tool"):
                results.append({**val, "interrupt_id": getattr(intr, "id", None)})
    return results


def find_pending_approval(graph, config) -> dict | None:
    """在图状态中查找第一个待审批的中断。"""
    pendings = find_all_pending_approvals(graph, config)
    if pendings:
        return {k: v for k, v in pendings[0].items() if k != "interrupt_id"}
    return None
