from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    TodoListMiddleware,
    ModelCallLimitMiddleware,
    dynamic_prompt,
)
from agent.config import Settings
from agent.llm.factory import build_llm
from agent.tools import get_tools
from agent.memory.checkpointer import build_checkpointer
from agent.memory.trimming import make_trim_middleware
from agent.memory.state_bar import make_state_bar_middleware
from agent.memory.user_memory import load_memories
from agent.prompts import SYSTEM_PROMPT


@dynamic_prompt
def build_prompt(request):
    configurable = (request.runtime.context or {}).get("configurable", {}) if request.runtime else {}
    user_id = configurable.get("user_id")
    memory = load_memories(user_id) if user_id else ""
    return f"{SYSTEM_PROMPT}\n\n{memory}" if memory else SYSTEM_PROMPT


def build_graph(settings: Settings | None = None):
    settings = settings or Settings()
    llm = build_llm(settings)
    cm = build_checkpointer(settings.sqlite_path)
    checkpointer = cm.__enter__()
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
    graph._checkpoint_cm = cm
    return graph
