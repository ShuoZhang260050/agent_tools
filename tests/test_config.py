import pytest
from agent.config import Settings

def test_settings_defaults_and_overrides():
    s = Settings(llm_api_key="sk-test", llm_base_url="http://localhost:11434/v1")
    assert s.llm_provider == "openai"
    assert s.llm_model == "gpt-4o-mini"
    assert s.llm_api_key.get_secret_value() == "sk-test"
    assert s.token_budget == 6000
    assert s.sqlite_path == "checkpoints.sqlite"

def test_settings_missing_api_key_raises():
    with pytest.raises(Exception):
        Settings()
