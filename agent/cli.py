import uuid

import click
from langchain_core.messages import HumanMessage

from agent.tools import get_tools
from agent.config import Settings


@click.group()
def main():
    """通用 Agent 框架 CLI。"""


@main.command()
def tools():
    """列出已注册工具。"""
    for t in get_tools():
        click.echo(f"- {t.name}: {t.description}")


@main.command()
@click.option("--thread", default=None, help="恢复指定会话 id")
def chat(thread):
    """交互式对话。"""
    from agent.graph import build_graph

    tid = thread or str(uuid.uuid4())
    click.echo(f"会话 id: {tid}（输入 /exit 退出）")
    graph = build_graph()
    config = {"configurable": {"thread_id": tid, "permission": "full_access"}}
    while True:
        msg = click.prompt("你", type=str, default="", show_default=False)
        if not msg or msg.strip() == "/exit":
            break
        for chunk, meta in graph.stream(
            {"messages": [HumanMessage(content=msg)]}, config, stream_mode="messages"
        ):
            if meta.get("langgraph_node") == "model" and chunk.content:
                click.echo(chunk.content, nl=False)
        click.echo()


@main.command()
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
def serve(host, port):
    """启动 HTTP API 服务。"""
    import uvicorn

    s = Settings()
    uvicorn.run("agent.api:app", host=host or s.api_host, port=port or s.api_port)


@main.command(name="cleanup-orphans")
def cleanup_orphans():
    """清理 writes/traces 中已不存在于 user_threads 的孤儿数据。"""
    from agent.memory.user_memory import cleanup_orphan_thread_data

    result = cleanup_orphan_thread_data(Settings().sqlite_path)
    for table, count in result.items():
        click.echo(f"{table}: 已清理 {count} 行孤儿数据")


if __name__ == "__main__":
    main()
