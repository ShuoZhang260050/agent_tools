from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    TodoListMiddleware,
    ModelCallLimitMiddleware,
    dynamic_prompt,
    wrap_model_call,
)
from agent.config import Settings
from agent.llm.factory import build_llm_with_model
from agent.tools import get_tools
from agent.memory.checkpointer import build_checkpointer
from agent.memory.trimming import make_trim_middleware
from agent.memory.state_bar import make_state_bar_middleware
from agent.memory.user_memory import load_memories, init_tables
from agent.memory.tracing import TracingCallbackHandler, init_traces_table
from agent.prompts import SYSTEM_PROMPT


@dynamic_prompt
def build_prompt(request):
    configurable = (request.runtime.context or {}).get("configurable", {}) if request.runtime else {}
    user_id = configurable.get("user_id")
    memory = load_memories(user_id) if user_id else ""
    return f"{SYSTEM_PROMPT}\n\n{memory}" if memory else SYSTEM_PROMPT


_checkpointers: dict[str, tuple[object, object]] = {}
_IMAGE_OMITTED = "[历史图片已省略：当前模型仅支持文本输入]"


def _vision_model_names(settings: Settings) -> set[str]:
    return {m.strip() for m in settings.vision_models.split(",") if m.strip()}


def _strip_image_parts(messages):
    stripped = []
    for msg in messages:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            stripped.append(msg)
            continue

        chunks = []
        omitted = 0
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    chunks.append(part)
            elif isinstance(part, dict):
                kind = part.get("type")
                if kind == "text":
                    text = part.get("text", "")
                    if text.strip():
                        chunks.append(text)
                elif kind in {"image_url", "image"}:
                    omitted += 1
                else:
                    omitted += 1

        if omitted:
            chunks.append(_IMAGE_OMITTED)
        new_content = "\n".join(chunks).strip() or _IMAGE_OMITTED
        if hasattr(msg, "model_copy"):
            stripped.append(msg.model_copy(update={"content": new_content}))
        else:
            stripped.append(msg.copy(update={"content": new_content}))
    return stripped


def make_text_only_input_middleware(settings: Settings, model_name: str):
    if model_name in _vision_model_names(settings):
        return None

    @wrap_model_call(name="TextOnlyInputMiddleware")
    def text_only_input(request, handler):
        return handler(request.override(messages=_strip_image_parts(request.messages)))

    return text_only_input


def _get_checkpointer(settings: Settings):
    path = settings.sqlite_path
    if path not in _checkpointers:
        cm = build_checkpointer(path)
        saver = cm.__enter__()
        _checkpointers[path] = (cm, saver)
    return _checkpointers[path][1]


def build_graph(settings: Settings | None = None):
    settings = settings or Settings()
    return build_graph_with_model(settings, settings.llm_model)


def build_graph_with_model(settings: Settings, model_name: str):
    init_tables()
    if settings.enable_tracing:
        init_traces_table()
    llm = build_llm_with_model(settings, model_name)
    checkpointer = _get_checkpointer(settings)
    callbacks = [TracingCallbackHandler(model_name)] if settings.enable_tracing else []
    middleware = [
        build_prompt,
        TodoListMiddleware(),
        ModelCallLimitMiddleware(run_limit=settings.model_call_limit),
    ]
    text_only_middleware = make_text_only_input_middleware(settings, model_name)
    if text_only_middleware is not None:
        middleware.append(text_only_middleware)
    middleware.extend([
        SummarizationMiddleware(
            model=llm,
            trigger=("messages", settings.summary_trigger_messages),
            keep=("messages", settings.summary_keep_messages),
        ),
        make_trim_middleware(settings.token_budget),
        make_state_bar_middleware(),
    ])
    graph = create_agent(
        model=llm,
        tools=get_tools(),
        middleware=middleware,
        checkpointer=checkpointer,
    )
    graph._tracing_callbacks = callbacks
    return graph
