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
