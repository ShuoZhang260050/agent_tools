from langgraph.checkpoint.sqlite import SqliteSaver

def build_checkpointer(sqlite_path: str):
    """返回 SqliteSaver 上下文管理器；用 with build_checkpointer(path) as saver:。"""
    return SqliteSaver.from_conn_string(sqlite_path)
