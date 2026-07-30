import random as _random
import uuid as _uuid
from langchain_core.tools import tool
from .registry import register


@register
@tool
def generate_uuid(version: int = 4) -> str:
    """生成 UUID。参数 version: 4（默认，随机）或 1（基于时间戳/MAC）。返回 UUID 字符串。"""
    if version == 1:
        return str(_uuid.uuid1())
    return str(_uuid.uuid4())


@register
@tool
def random_int(min_val: int, max_val: int) -> str:
    """生成指定范围内的随机整数（含两端）。参数 min_val: 最小值, max_val: 最大值。"""
    if min_val > max_val:
        return f"错误：min_val({min_val}) 大于 max_val({max_val})。"
    return str(_random.randint(min_val, max_val))


@register
@tool
def random_choice(items: str) -> str:
    """从列表中随机选择一个元素。参数 items: 逗号分隔的列表，如 '苹果,香蕉,橙子'。"""
    candidates = [x.strip() for x in items.split(",") if x.strip()]
    if not candidates:
        return "错误：列表为空。"
    return _random.choice(candidates)
