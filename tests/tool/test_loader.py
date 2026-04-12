from __future__ import annotations

from pathlib import Path

from runweave.tool.loader import ToolLoader


def _write_tool_file(tools_dir: Path, filename: str, content: str) -> None:
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / filename).write_text(content, encoding="utf-8")


def test_scan_tool_subclass(tmp_path: Path) -> None:
    """Tool subclass instantiated at module level should be discovered."""
    _write_tool_file(
        tmp_path,
        "greet.py",
        """\
from smolagents import Tool

class GreetTool(Tool):
    name = "greet"
    description = "Say hello"
    inputs = {"name": {"type": "string", "description": "Who to greet"}}
    output_type = "string"

    def forward(self, name: str) -> str:
        return f"Hello, {name}!"

greet = GreetTool()
""",
    )
    loader = ToolLoader(tmp_path)
    assert "greet" in loader.list_names()
    tools = loader.get_tools()
    assert len(tools) == 1
    assert tools[0].name == "greet"


def test_scan_multiple_tools(tmp_path: Path) -> None:
    """Tools from multiple files should all be discovered."""
    _write_tool_file(
        tmp_path,
        "alpha.py",
        """\
from smolagents import Tool

class AlphaTool(Tool):
    name = "alpha"
    description = "Alpha tool"
    inputs = {}
    output_type = "string"
    def forward(self) -> str:
        return "a"

alpha = AlphaTool()
""",
    )
    _write_tool_file(
        tmp_path,
        "beta.py",
        """\
from smolagents import Tool

class BetaTool(Tool):
    name = "beta"
    description = "Beta tool"
    inputs = {}
    output_type = "string"
    def forward(self) -> str:
        return "b"

beta = BetaTool()
""",
    )
    loader = ToolLoader(tmp_path)
    names = loader.list_names()
    assert "alpha" in names
    assert "beta" in names
    assert len(loader.get_tools()) == 2


def test_get_tools_by_name(tmp_path: Path) -> None:
    """Filtering by name should return only the specified tools."""
    _write_tool_file(
        tmp_path,
        "tools.py",
        """\
from smolagents import Tool

class OneTool(Tool):
    name = "one"
    description = "First"
    inputs = {}
    output_type = "string"
    def forward(self) -> str:
        return "1"

class TwoTool(Tool):
    name = "two"
    description = "Second"
    inputs = {}
    output_type = "string"
    def forward(self) -> str:
        return "2"

one = OneTool()
two = TwoTool()
""",
    )
    loader = ToolLoader(tmp_path)
    selected = loader.get_tools(["one"])
    assert len(selected) == 1
    assert selected[0].name == "one"


def test_get_tools_unknown_name_ignored(tmp_path: Path) -> None:
    """Requesting a nonexistent tool name should be silently ignored."""
    _write_tool_file(
        tmp_path,
        "tools.py",
        """\
from smolagents import Tool

class OneTool(Tool):
    name = "one"
    description = "First"
    inputs = {}
    output_type = "string"
    def forward(self) -> str:
        return "1"

one = OneTool()
""",
    )
    loader = ToolLoader(tmp_path)
    selected = loader.get_tools(["one", "nonexistent"])
    assert len(selected) == 1


def test_empty_dir(tmp_path: Path) -> None:
    """Empty directory should not cause errors."""
    tmp_path.mkdir(exist_ok=True)
    loader = ToolLoader(tmp_path)
    assert loader.list_names() == []
    assert loader.get_tools() == []


def test_nonexistent_dir(tmp_path: Path) -> None:
    """Nonexistent directory should not cause errors."""
    loader = ToolLoader(tmp_path / "no_such_dir")
    assert loader.list_names() == []


def test_underscore_files_skipped(tmp_path: Path) -> None:
    """Files starting with _ should be skipped during scanning."""
    _write_tool_file(
        tmp_path,
        "_private.py",
        """\
from smolagents import Tool

class HiddenTool(Tool):
    name = "hidden"
    description = "Should not be found"
    inputs = {}
    output_type = "string"
    def forward(self) -> str:
        return "x"

hidden = HiddenTool()
""",
    )
    loader = ToolLoader(tmp_path)
    assert "hidden" not in loader.list_names()


def test_broken_file_records_error(tmp_path: Path) -> None:
    """Files with syntax errors should record an error without affecting other tools."""
    _write_tool_file(tmp_path, "broken.py", "this is not valid python!!!")
    _write_tool_file(
        tmp_path,
        "good.py",
        """\
from smolagents import Tool

class GoodTool(Tool):
    name = "good"
    description = "Works fine"
    inputs = {}
    output_type = "string"
    def forward(self) -> str:
        return "ok"

good = GoodTool()
""",
    )
    loader = ToolLoader(tmp_path)
    assert "good" in loader.list_names()
    assert "broken.py" in loader._errors


def test_catalog_output(tmp_path: Path) -> None:
    """get_catalog should produce a readable catalog text."""
    _write_tool_file(
        tmp_path,
        "calc.py",
        """\
from smolagents import Tool

class CalcTool(Tool):
    name = "calculator"
    description = "Perform arithmetic"
    inputs = {"expr": {"type": "string", "description": "expression"}}
    output_type = "string"
    def forward(self, expr: str) -> str:
        return str(eval(expr))

calculator = CalcTool()
""",
    )
    loader = ToolLoader(tmp_path)
    catalog = loader.get_catalog()
    assert "calculator" in catalog
    assert "Perform arithmetic" in catalog


def test_catalog_empty_when_no_tools(tmp_path: Path) -> None:
    """Catalog should be an empty string when no tools are found."""
    loader = ToolLoader(tmp_path)
    assert loader.get_catalog() == ""
