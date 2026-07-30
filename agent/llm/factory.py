from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from agent.config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    return build_llm_with_model(settings, settings.llm_model)


def build_llm_with_model(settings: Settings, model_name: str) -> BaseChatModel:
    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=model_name,
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
        )
    raise ValueError(f"未知 LLM provider: {settings.llm_provider}")
