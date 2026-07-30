from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    TodoListMiddleware,
    ModelCallLimitMiddleware,
    dynamic_prompt,
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
    callbacks = [TracingCallbackHandler()] if settings.enable_tracing else []
    graph = create_agent(
        model=llm,
        tools=get_tools(),
        middleware=[
            build_prompt,
            TodoListMiddleware(),
            ModelCallLimitMiddleware(run_limit=settings.model_call_limit),
            SummarizationMiddleware(
                model=llm,
                trigger=("messages", settings.summary_trigger_messages),
                keep=("messages", settings.summary_keep_messages),
            ),
            make_trim_middleware(settings.token_budget),
            make_state_bar_middleware(),
        ],
        checkpointer=checkpointer,
    )
    graph._tracing_callbacks = callbacks
    return graph
