from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from agent.config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
        )
    raise ValueError(f"未知 LLM provider: {settings.llm_provider}")
