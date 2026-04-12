from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from smolagents import Tool


class ToolLoader:
    """Scan a tools directory and dynamically load user-defined Tools.

    Supports two definition patterns:
    - @tool decorator (produces a Tool instance automatically)
    - Tool subclass (must be instantiated at module level)

    ToolLoader only collects module-level Tool instances; it does not
    auto-instantiate classes.
    """

    def __init__(self, tools_dir: Path) -> None:
        self.tools_dir = tools_dir.resolve()
        self._registry: dict[str, Tool] = {}
        self._errors: dict[str, str] = {}
        self._scan()

    # -- Public methods ------------------------------------------------

    def list_names(self) -> list[str]:
        """Return the names of all discovered tools."""
        return list(self._registry.keys())

    def get_tools(self, names: list[str] | None = None) -> list[Tool]:
        """Return Tool instances by name. Returns all when names is None."""
        if names is None:
            return list(self._registry.values())
        result: list[Tool] = []
        for name in names:
            if name in self._registry:
                result.append(self._registry[name])
        return result

    def get_catalog(self) -> str:
        """Generate a tool catalog string for injection into the system prompt."""
        if not self._registry:
            return ""
        lines = ["## Available Tools (custom)"]
        for name, t in self._registry.items():
            desc = getattr(t, "description", "") or ""
            lines.append(f"- **{name}**: {desc}")
        return "\n".join(lines)

    # -- Internal methods ----------------------------------------------

    def _scan(self) -> None:
        """Scan all .py files under tools_dir and discover Tool instances."""
        if not self.tools_dir.is_dir():
            return
        for py_file in sorted(self.tools_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            self._load_module(py_file)

    def _load_module(self, path: Path) -> None:
        """Import a single .py file and register all Tool instances found."""
        module_name = f"_runweave_tools_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            self._errors[path.name] = "cannot create module spec"
            return
        module = importlib.util.module_from_spec(spec)
        # Temporarily add to sys.modules so intra-module imports work
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
