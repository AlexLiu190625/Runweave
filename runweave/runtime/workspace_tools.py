"""Workspace file I/O tools — safe, sandboxed file access for agents."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from smolagents import Tool

MAX_READ_BYTES = 102_400  # 100 KB
MAX_WRITE_BYTES = 1_048_576  # 1 MB
MAX_LIST_ENTRIES = 200
MAX_PATH_DEPTH = 15  # maximum directory nesting depth


def _resolve_safe(workspace: Path, user_path: str) -> Path:
    """Resolve *user_path* relative to *workspace* with security checks.

    Raises ``ValueError`` on absolute paths, null bytes, excessive depth,
    or any path that resolves outside the workspace (including ``..``
    traversal and symlink escaping).
    """
    if PurePosixPath(user_path).is_absolute():
        raise ValueError("Absolute paths are not allowed.")
    if "\x00" in user_path:
        raise ValueError("Null bytes are not allowed in file paths.")
    parts = PurePosixPath(user_path).parts
    if len(parts) > MAX_PATH_DEPTH:
        raise ValueError(
            f"Path too deep ({len(parts)} levels, max {MAX_PATH_DEPTH})."
        )
    resolved = (workspace / user_path).resolve()
    if not resolved.is_relative_to(workspace.resolve()):
        raise ValueError(
            "Path traversal detected — access outside the workspace is not allowed."
        )
    return resolved


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class WriteFileTool(Tool):
    """Create or overwrite a file inside the agent's workspace."""

    name = "write_file"
    description = (
        "Create or overwrite a file in the workspace. "
        "The path must be relative (e.g. 'src/main.py'). "
        "Parent directories are created automatically."
    )
    inputs = {
        "path": {"type": "string", "description": "Relative file path"},
        "content": {"type": "string", "description": "File content to write"},
    }
    output_type = "string"

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir.resolve()
        super().__init__()

    def forward(self, path: str, content: str) -> str:
        try:
            resolved = _resolve_safe(self.workspace_dir, path)
        except ValueError as exc:
            return f"Error: {exc}"
        byte_len = len(content.encode("utf-8"))
        if byte_len > MAX_WRITE_BYTES:
            return f"Error: content size ({byte_len} bytes) exceeds the {MAX_WRITE_BYTES} byte limit."
        resolved.parent.mkdir(parents=True, exist_ok=True)
        # Re-check after mkdir: verify parent hasn't become a symlink
        # pointing outside the workspace (TOCTOU defence).
        real_parent = resolved.parent.resolve()
        if not real_parent.is_relative_to(self.workspace_dir):
            return "Error: write target resolves outside the workspace."
        resolved.write_text(content, encoding="utf-8")
        return f"Written {byte_len} bytes to {path}"


class ReadFileTool(Tool):
    """Read the contents of a file inside the agent's workspace."""

    name = "read_file"
    description = (
        "Read the contents of a file in the workspace. "
        "The path must be relative. Large files are truncated."
    )
    inputs = {
        "path": {"type": "string", "description": "Relative file path"},
    }
    output_type = "string"

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir.resolve()
        super().__init__()

    def forward(self, path: str) -> str:
        try:
            resolved = _resolve_safe(self.workspace_dir, path)
        except ValueError as exc:
            return f"Error: {exc}"
        if not resolved.is_file():
            return f"Error: file not found: {path}"
        size = resolved.stat().st_size
        data = resolved.read_bytes()[:MAX_READ_BYTES]
        text = data.decode("utf-8", errors="replace")
        if size > MAX_READ_BYTES:
            text += f"\n\n[Truncated: showing first {MAX_READ_BYTES} of {size} bytes]"
        return text


class ListFilesTool(Tool):
    """List files and directories inside the agent's workspace."""

    name = "list_files"
    description = (
        "List files and directories in the workspace. "
        "Pass a relative subdirectory path, or omit to list the workspace root."
    )
    inputs = {
        "path": {
            "type": "string",
            "description": "Relative subdirectory path (empty or omit for workspace root)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir.resolve()
        super().__init__()

    def forward(self, path: str | None = None) -> str:
        if path:
            try:
                target = _resolve_safe(self.workspace_dir, path)
            except ValueError as exc:
                return f"Error: {exc}"
        else:
            target = self.workspace_dir
        if not target.is_dir():
            return f"Error: not a directory: {path}"
        entries = sorted(target.iterdir())[:MAX_LIST_ENTRIES]
        if not entries:
            return "(empty directory)"
        lines = []
        for entry in entries:
            prefix = "d " if entry.is_dir() else "f "
            lines.append(prefix + entry.relative_to(self.workspace_dir).as_posix())
        return "\n".join(lines)
