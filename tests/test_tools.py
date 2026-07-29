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
    assert "<external_content" in out

def test_web_search_empty():
    p = _patch_ddgs([])
    try:
        assert "无" in web_search.invoke({"query": "zzz"})
    finally:
        p.stop()

def test_web_search_handles_error():
    inst = MagicMock()
    inst.text.side_effect = RuntimeError("boom")
    patcher = patch("agent.tools.web_search.DDGS")
    m = patcher.start()
    m.return_value.__enter__.return_value = inst
    try:
        out = web_search.invoke({"query": "anything"})
    finally:
        patcher.stop()
    assert "搜索失败" in out


import json
from agent.tools.weather import weather

_WTTR_J1 = {
    "current_condition": [{
        "temp_C": "31",
        "FeelsLikeC": "35",
        "humidity": "61",
        "weatherDesc": [{"value": "Sunny"}],
        "windspeedKmph": "4",
        "winddir16Point": "S",
    }],
    "nearest_area": [{"areaName": [{"value": "Beijing"}]}],
}


def _mock_urlopen(data):
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__.return_value = resp  # with urlopen(...) as r: 时让 r 就是 resp 本身
    return patch("agent.tools.weather.urlopen", return_value=resp)


def test_weather_returns_conditions():
    p = _mock_urlopen(_WTTR_J1)
    p.start()
    try:
        captured = {}
        real_quote = __import__("urllib.parse", fromlist=["quote"]).quote

        def spy_quote(loc):
            captured["location"] = loc
            return real_quote(loc)

        with patch("agent.tools.weather.quote", side_effect=spy_quote):
            out = weather.invoke({"location": "北京"})
    finally:
        p.stop()
    assert "31" in out and "35" in out and "Sunny" in out
    assert "61" in out
    assert "<external_content" in out
    assert captured["location"] == "北京"  # 验证中文地名原样传给 quote 编码


def test_weather_handles_error():
    with patch("agent.tools.weather.urlopen", side_effect=RuntimeError("boom")):
        out = weather.invoke({"location": "nowhere"})
    assert "失败" in out


def test_weather_handles_missing_fields():
    p = _mock_urlopen({"current_condition": [], "nearest_area": []})
    p.start()
    try:
        out = weather.invoke({"location": "未知地"})
    finally:
        p.stop()
    assert "未知" in out or "失败" in out or "?" in out
