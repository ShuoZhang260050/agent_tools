from typing import Any

# 模块级工具注册表，所有工具通过 register 装饰器登记于此
_REGISTRY: list[Any] = []

def register(tool: Any) -> Any:
    """注册工具到全局注册表。"""
    # 将工具追加到注册表并原样返回，支持 @register 装饰器用法
    _REGISTRY.append(tool)
    return tool

def get_tools() -> list[Any]:
    """获取已注册工具列表。"""
    # 返回注册表副本，防止调用方意外修改内部状态
    return list(_REGISTRY)
