"""Tests for workspace file I/O tools."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from runweave.runtime.workspace_tools import (
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
    _resolve_safe,
)


# ---------------------------------------------------------------------------
# _resolve_safe
# ---------------------------------------------------------------------------


class TestResolveSafe:
    def test_simple_relative(self, tmp_path: Path) -> None:
        result = _resolve_safe(tmp_path, "hello.txt")
        assert result == (tmp_path / "hello.txt").resolve()

    def test_nested_relative(self, tmp_path: Path) -> None:
        result = _resolve_safe(tmp_path, "src/main.py")
        assert result == (tmp_path / "src/main.py").resolve()

    def test_rejects_absolute(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Absolute paths"):
            _resolve_safe(tmp_path, "/etc/passwd")

    def test_rejects_dotdot_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="traversal"):
            _resolve_safe(tmp_path, "../../etc/passwd")

    def test_rejects_intermediate_dotdot(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="traversal"):
            _resolve_safe(tmp_path, "sub/../../etc/passwd")

    def test_rejects_null_byte(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Null bytes"):
            _resolve_safe(tmp_path, "hello\x00.txt")

    def test_rejects_deep_path(self, tmp_path: Path) -> None:
        deep = "/".join(f"d{i}" for i in range(20)) + "/file.txt"
        with pytest.raises(ValueError, match="too deep"):
            _resolve_safe(tmp_path, deep)

    def test_rejects_symlink_escape(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        link = workspace / "escape"
        link.symlink_to(outside)
        with pytest.raises(ValueError, match="traversal"):
            _resolve_safe(workspace, "escape/secret.txt")


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------


class TestWriteFileTool:
    def test_write_simple(self, tmp_path: Path) -> None:
        tool = WriteFileTool(tmp_path)
        result = tool.forward("hello.txt", "hello world")
        assert "Written" in result
        assert (tmp_path / "hello.txt").read_text() == "hello world"

    def test_write_creates_subdirs(self, tmp_path: Path) -> None:
        tool = WriteFileTool(tmp_path)
        tool.forward("a/b/c.py", "print('ok')")
        assert (tmp_path / "a/b/c.py").read_text() == "print('ok')"

    def test_write_rejects_traversal(self, tmp_path: Path) -> None:
        tool = WriteFileTool(tmp_path)
        result = tool.forward("../../evil.txt", "pwned")
        assert "Error" in result
        assert not (tmp_path / "../../evil.txt").resolve().exists()

    def test_write_rejects_absolute(self, tmp_path: Path) -> None:
        tool = WriteFileTool(tmp_path)
        result = tool.forward("/tmp/evil.txt", "pwned")
        assert "Error" in result

    def test_write_rejects_oversized(self, tmp_path: Path) -> None:
        tool = WriteFileTool(tmp_path)
        big_content = "x" * (1_048_576 + 1)
        result = tool.forward("big.txt", big_content)
        assert "Error" in result
        assert not (tmp_path / "big.txt").exists()

    def test_write_overwrite(self, tmp_path: Path) -> None:
        tool = WriteFileTool(tmp_path)
        tool.forward("f.txt", "v1")
        tool.forward("f.txt", "v2")
        assert (tmp_path / "f.txt").read_text() == "v2"


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------


class TestReadFileTool:
    def test_read_existing(self, tmp_path: Path) -> None:
        (tmp_path / "data.txt").write_text("hello")
        tool = ReadFileTool(tmp_path)
        assert tool.forward("data.txt") == "hello"

    def test_read_not_found(self, tmp_path: Path) -> None:
        tool = ReadFileTool(tmp_path)
        result = tool.forward("nope.txt")
        assert "Error" in result and "not found" in result

    def test_read_truncates_large_file(self, tmp_path: Path) -> None:
        big = "a" * 200_000
        (tmp_path / "big.txt").write_text(big)
        tool = ReadFileTool(tmp_path)
        result = tool.forward("big.txt")
        assert "[Truncated" in result
        assert len(result) < 200_000

    def test_read_rejects_traversal(self, tmp_path: Path) -> None:
        tool = ReadFileTool(tmp_path)
        result = tool.forward("../../etc/passwd")
        assert "Error" in result


# ---------------------------------------------------------------------------
# ListFilesTool
# ---------------------------------------------------------------------------


class TestListFilesTool:
    def test_list_root(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").touch()
        (tmp_path / "b.py").touch()
        (tmp_path / "sub").mkdir()
        tool = ListFilesTool(tmp_path)
        result = tool.forward()
        assert "f a.py" in result
        assert "f b.py" in result
        assert "d sub" in result

    def test_list_subdir(self, tmp_path: Path) -> None:
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").touch()
        tool = ListFilesTool(tmp_path)
        result = tool.forward("src")
        assert "f src/main.py" in result

    def test_list_empty(self, tmp_path: Path) -> None:
        tool = ListFilesTool(tmp_path)
        assert tool.forward() == "(empty directory)"

    def test_list_none_path(self, tmp_path: Path) -> None:
        (tmp_path / "x.txt").touch()
        tool = ListFilesTool(tmp_path)
        result = tool.forward(None)
        assert "f x.txt" in result

    def test_list_rejects_traversal(self, tmp_path: Path) -> None:
        tool = ListFilesTool(tmp_path)
        result = tool.forward("../../")
        assert "Error" in result
