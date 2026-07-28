from agent.tools.registry import register, get_tools

def test_register_and_get_tools():
    before = len(get_tools())
    @register
    def fake_tool(x: str) -> str:
        """fake"""
        return x
    assert fake_tool in get_tools()
    assert len(get_tools()) == before + 1

def test_get_tools_returns_copy():
    tools = get_tools()
    tools.append("mutated")
    assert "mutated" not in get_tools()
