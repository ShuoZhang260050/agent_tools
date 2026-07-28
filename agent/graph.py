from langchain.agents import create_agent
from agent.config import Settings
from agent.llm.factory import build_llm
from agent.tools import get_tools
from agent.memory.checkpointer import build_checkpointer
from agent.memory.trimming import make_trim_middleware
from agent.prompts import SYSTEM_PROMPT


def build_graph(settings: Settings | None = None):
    settings = settings or Settings()
    llm = build_llm(settings)
    cm = build_checkpointer(settings.sqlite_path)
    checkpointer = cm.__enter__()
    graph = create_agent(
        model=llm,
        tools=get_tools(),
        system_prompt=SYSTEM_PROMPT,
        middleware=[make_trim_middleware(settings.token_budget)],
        checkpointer=checkpointer,
    )
    # 持有 checkpointer 上下文管理器，防止其生成器被 GC 回收后关闭 sqlite 连接
    graph._checkpoint_cm = cm
    return graph
