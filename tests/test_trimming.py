from agent.memory.trimming import make_trim_middleware
from agent.prompts import SYSTEM_PROMPT

def test_system_prompt_is_str():
    assert isinstance(SYSTEM_PROMPT, str) and SYSTEM_PROMPT

def test_trim_middleware_factory():
    mw = make_trim_middleware(max_tokens=20)
    assert mw is not None
