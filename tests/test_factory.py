import pytest
from agent.config import Settings
from agent.llm.factory import build_llm

def _s(**kw):
    base = dict(llm_api_key="sk-x", llm_model="gpt-4o-mini")
    base.update(kw)
    return Settings(**base)

def test_build_llm_openai_attrs():
    llm = build_llm(_s(llm_base_url="http://localhost:11434/v1"))
    assert llm.model_name == "gpt-4o-mini"
    assert str(llm.openai_api_base).rstrip("/") == "http://localhost:11434/v1"

def test_build_llm_unknown_provider():
    with pytest.raises(ValueError):
        build_llm(_s(llm_provider="bogus"))
