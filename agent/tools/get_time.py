from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from langchain_core.tools import tool
from .registry import register


@register
@tool
def get_time(timezone: str = "Asia/Shanghai") -> str:
    """获取指定时区的当前时间。
    输入 IANA 时区名，如 Asia/Shanghai、America/New_York、Europe/London、Asia/Tokyo。
    返回日期、时间、星期和 UTC 偏移量。"""
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return f"未知时区：{timezone}。请使用 IANA 时区名，如 Asia/Shanghai、America/New_York。"
    now = datetime.now(tz)
    offset = now.strftime("%z")
    offset_fmt = f"UTC{'+' if offset[0] != '-' else ''}{offset[:3]}:{offset[3:]}"
    return (
        f'<external_content source="get_time">\n'
        f'{timezone}\n'
        f'{now.strftime("%Y-%m-%d %H:%M:%S %A")}\n'
        f'{offset_fmt}\n'
        f'</external_content>'
    )


@register
@tool
def convert_time(datetime_str: str, from_timezone: str, to_timezone: str) -> str:
    """将时间从一个时区转换到另一个时区。
    参数:
        datetime_str: 时间字符串，格式 YYYY-MM-DD HH:MM，如 "2024-01-15 14:30"
        from_timezone: 源时区 IANA 名，如 Asia/Shanghai
        to_timezone: 目标时区 IANA 名，如 America/New_York
    """
    try:
        src_tz = ZoneInfo(from_timezone)
        dst_tz = ZoneInfo(to_timezone)
    except ZoneInfoNotFoundError as e:
        return f"未知时区：{e}。请使用 IANA 时区名。"
    try:
        naive = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
    except ValueError:
        return f"时间格式错误：{datetime_str}。请用 YYYY-MM-DD HH:MM 格式，如 2024-01-15 14:30。"
    src_dt = naive.replace(tzinfo=src_tz)
    dst_dt = src_dt.astimezone(dst_tz)
    return (
        f'<external_content source="convert_time">\n'
        f'{from_timezone}: {src_dt.strftime("%Y-%m-%d %H:%M %A")}\n'
        f'{to_timezone}: {dst_dt.strftime("%Y-%m-%d %H:%M %A")}\n'
        f'</external_content>'
    )
