from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """pydantic-settings 配置类，读取 .env 环境变量。"""

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

    @field_validator("jwt_secret")
    @classmethod
    def _validate_jwt_secret(cls, v: str) -> str:
        """校验 JWT 密钥长度，HS256 安全下限 32 字符。"""
        if len(v) < 32:
            raise ValueError("JWT_SECRET 长度不足 32 字符，HS256 不安全。请在 .env 中设置更长的密钥。")
        return v

    token_expire_hours: int = 168
    sqlite_path: str = "checkpoints.sqlite"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 50
    rag_top_k: int = 3
    enable_tracing: bool = True
    available_models: str = ""
    vision_models: str = ""
    run_python_timeout: int = 120
    run_command_timeout: int = 300

    # Shadow Workspace 沙箱配置
    # shadow 副本总大小上限（字节），超过则拒绝创建。默认 200MB
    shadow_max_bytes: int = 200 * 1024 * 1024
    # shadow 拷贝时跳过的目录名（逗号分隔），这些目录不会被复制到 shadow
    shadow_skip_dirs: str = ".git,node_modules,.venv,__pycache__,.pytest_cache,dist,build,.superpowers"
    # shadow 中验证命令的默认超时秒数
    shadow_verify_timeout: int = 120

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")
