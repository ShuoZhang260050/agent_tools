from datetime import datetime
from zoneinfo import ZoneInfo
from agent.tools.get_time import get_time, convert_time


def test_get_time_default():
    out = get_time.invoke({"timezone": "Asia/Shanghai"})
    assert "<external_content" in out
    assert "Asia/Shanghai" in out
    assert "UTC" in out


def test_get_time_specific_tz():
    out = get_time.invoke({"timezone": "America/New_York"})
    assert "America/New_York" in out
    assert "<external_content" in out


def test_get_time_bad_tz():
    out = get_time.invoke({"timezone": "Mars/Olympus"})
    assert "未知时区" in out


def test_convert_time_basic():
    out = convert_time.invoke({
        "datetime_str": "2024-07-15 14:30",
        "from_timezone": "Asia/Shanghai",
        "to_timezone": "America/New_York",
    })
    assert "<external_content" in out
    assert "Asia/Shanghai" in out
    assert "America/New_York" in out
    assert "2024-07-15 14:30" in out
    # Shanghai is UTC+8, NY in July is UTC-4 (EDT), so 14:30 -> 02:30
    assert "02:30" in out


def test_convert_time_bad_format():
    out = convert_time.invoke({
        "datetime_str": "not-a-date",
        "from_timezone": "Asia/Shanghai",
        "to_timezone": "America/New_York",
    })
    assert "时间格式错误" in out


def test_convert_time_bad_tz():
    out = convert_time.invoke({
        "datetime_str": "2024-07-15 14:30",
        "from_timezone": "Asia/Shanghai",
        "to_timezone": "Galaxy/Andromeda",
    })
    assert "未知时区" in out
