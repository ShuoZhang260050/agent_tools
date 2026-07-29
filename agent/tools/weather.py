import json
from urllib.parse import quote
from urllib.request import urlopen
from langchain_core.tools import tool
from .registry import register


@register
@tool
def weather(location: str) -> str:
    """查询指定地点的当前天气。输入城市名（支持中文，如"北京"、"上海"、"Shanghai"）。返回温度、体感温度、天气描述、湿度与风速。"""
    url = f"https://wttr.in/{quote(location)}?format=j1&lang=zh-cn"
    try:
        with urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return f"天气查询失败：{type(e).__name__}。可能地名无效或网络异常，请稍后重试。"

    conds = data.get("current_condition") or []
    areas = data.get("nearest_area") or []
    cond = conds[0] if conds else {}
    area_name = ((areas[0].get("areaName") if areas else None) or [{"value": location}])[0].get("value", location)
    desc = (((cond.get("lang_zh-cn") or cond.get("weatherDesc")) or [{"value": ""}])[0].get("value")) or "未知"

    return (
        '<external_content source="weather">\n'
        + "\n".join([
            f"{area_name}({location}) 当前天气",
            f"温度: {cond.get('temp_C', '?')}°C (体感 {cond.get('FeelsLikeC', '?')}°C)",
            f"天气: {desc}",
            f"湿度: {cond.get('humidity', '?')}%",
            f"风速: {cond.get('windspeedKmph', '?')} km/h 风向: {cond.get('winddir16Point', '?')}",
        ])
        + "\n</external_content>"
    )
