from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from agent.memory.user_memory import set_memory


class SaveMemoryTool(BaseTool):
    """用户记忆保存工具。"""
    name: str = "save_memory"
    description: str = (
        "记住用户提供的偏好、事实或指令，供后续对话使用。"
        "key 是简短标签（如'语言'、'工作'），value 是具体内容。"
    )

    def _run(self, key: str, value: str, config: RunnableConfig) -> str:
        """保存用户记忆键值对。"""
        user_id = (config or {}).get("configurable", {}).get("user_id")
        if not user_id:
            return "无法保存：未识别用户身份"
        return set_memory(user_id, key, value)


save_memory = SaveMemoryTool()
