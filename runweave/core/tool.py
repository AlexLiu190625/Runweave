from typing import Callable
#显示定义工具类，用于定义工具的名称、描述和函数


class Tool:
    #初始化工具的名称、描述和函数
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    #定义工具的调用方式
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)