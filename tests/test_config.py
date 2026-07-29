import pytest
from agent.config import Settings


def test_settings_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("SQLITE_PATH", raising=False)
    s = Settings(_env_file=None, llm_api_key="sk-test",
                 llm_base_url="http://localhost:11434/v1")
    assert s.llm_provider == "openai"
    assert s.llm_model == "gpt-4o-mini"
    assert s.llm_api_key.get_secret_value() == "sk-test"
    assert s.llm_base_url == "http://localhost:11434/v1"
    assert s.token_budget == 500000
    assert s.sqlite_path == "checkpoints.sqlite"


def test_settings_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(Exception):
        Settings(_env_file=None)
