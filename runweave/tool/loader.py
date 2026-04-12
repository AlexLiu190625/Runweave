from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from smolagents import Tool


class ToolLoader:
    """扫描 tools 目录，动态加载用户自定义 Tool。

    支持两种定义方式：
    - @tool 装饰器（自动生成 Tool 实例）
    - Tool 子类（需在模块中实例化）

    ToolLoader 只收集模块级的 Tool 实例，不自动实例化类。
    """

    def __init__(self, tools_dir: Path) -> None:
        self.tools_dir = tools_dir.resolve()
        self._registry: dict[str, Tool] = {}
        self._errors: dict[str, str] = {}
        self._scan()

    # ── 公开方法 ──────────────────────────────────────────

    def list_names(self) -> list[str]:
        """返回所有已发现 tool 的名称。"""
        return list(self._registry.keys())

    def get_tools(self, names: list[str] | None = None) -> list[Tool]:
        """按名称返回 Tool 实例。names=None 时返回全部。"""
        if names is None:
            return list(self._registry.values())
        result: list[Tool] = []
        for name in names:
            if name in self._registry:
                result.append(self._registry[name])
        return result

    def get_catalog(self) -> str:
        """生成 tool 目录文本，用于注入 system prompt。"""
        if not self._registry:
            return ""
        lines = ["## Available Tools (custom)"]
        for name, t in self._registry.items():
            desc = getattr(t, "description", "") or ""
            lines.append(f"- **{name}**: {desc}")
        return "\n".join(lines)

    # ── 内部方法 ──────────────────────────────────────────

    def _scan(self) -> None:
        """扫描 tools_dir 下所有 .py 文件，导入并发现 Tool 实例。"""
        if not self.tools_dir.is_dir():
            return
        for py_file in sorted(self.tools_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            self._load_module(py_file)

    def _load_module(self, path: Path) -> None:
        """导入单个 .py 文件，注册其中所有 Tool 实例。"""
        module_name = f"_runweave_tools_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            self._errors[path.name] = "cannot create module spec"
            return
        module = importlib.util.module_from_spec(spec)
        # 临时加入 sys.modules，防止模块内部 import 出错
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            self._errors[path.name] = str(exc)
            sys.modules.pop(module_name, None)
            return

        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            obj = getattr(module, attr_name)
            if isinstance(obj, Tool) and hasattr(obj, "name") and obj.name:
                self._registry[obj.name] = obj
