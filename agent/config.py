from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: SecretStr
    llm_base_url: str | None = None
    llm_temperature: float = 0.7
    token_budget: int = 500000
    model_call_limit: int = 25
    summary_trigger_messages: int = 30
    summary_keep_messages: int = 10
    jwt_secret: str = "change-me-in-production"
    token_expire_hours: int = 168
    sqlite_path: str = "checkpoints.sqlite"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")
