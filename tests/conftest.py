from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeToolModel(BaseChatModel):
    """测试用：按队列返回消息；bind_tools 返回自身。"""
    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)

    def bind_tools(self, tools, **kwargs):
        return self

    def get_num_tokens_from_messages(self, messages):
        # 测试替身：用字符数近似 token 数，避免依赖真实 tokenizer
        return sum(len(m.content) if isinstance(m.content, str) else 1 for m in messages)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._responses.pop(0) if self._responses else AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _identifying_params(self):
        return {"name": "FakeToolModel"}

    @property
    def _llm_type(self):
        return "fake-tool"
