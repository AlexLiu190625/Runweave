from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Thread:
    """An independent unit of work with its own workspace and memory archive."""

    # Unique identifier
    id: str
    # Creation time (ISO format)
    created_at: str
    # Workspace directory where the agent reads and writes files
    workspace_dir: Path
    # Path to the serialized memory file
    memory_path: Path
    # Path to the summary file
    summary_path: Path
    # Path to HISTORY.md (structured run history)
    history_path: Path
    # Path to the runs/ directory (per-run detailed records)
    runs_dir: Path
