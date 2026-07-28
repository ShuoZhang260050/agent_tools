import pytest
from agent.tools.calculator import calculator

def test_calc_basic():
    assert calculator.invoke({"expression": "2 + 3"}) == "5"
def test_calc_precedence():
    assert calculator.invoke({"expression": "2 * 3 + 4"}) == "10"
    assert calculator.invoke({"expression": "10 / 4"}) == "2.5"
def test_calc_rejects_dangerous():
    for bad in ["__import__('os')", "open('a')", "1 and 2"]:
        with pytest.raises(Exception):
            calculator.invoke({"expression": bad})

from unittest.mock import patch, MagicMock
from agent.tools.web_search import web_search

def _patch_ddgs(results):
    inst = MagicMock()
    inst.text.return_value = results
    patcher = patch("agent.tools.web_search.DDGS")
    m = patcher.start()
    m.return_value.__enter__.return_value = inst
    return patcher

def test_web_search_formats_results():
    p = _patch_ddgs([{"title": "T1", "body": "B1"}, {"title": "T2", "body": "B2"}])
    try:
        out = web_search.invoke({"query": "python"})
    finally:
        p.stop()
    assert "T1" in out and "B1" in out and "T2" in out

def test_web_search_empty():
    p = _patch_ddgs([])
    try:
        assert "无" in web_search.invoke({"query": "zzz"})
    finally:
        p.stop()
