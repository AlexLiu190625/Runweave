from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents import AgentMemory


def save_memory(memory: AgentMemory, path: Path) -> None:
    """Serialize AgentMemory to a JSON file.

    Uses get_succinct_steps() for a compact format (excludes
    model_input_messages). Serialization is handled by smolagents'
    internal .dict() method. The saved data is for user inspection
    only and is never fed back into the agent on subsequent runs.
    """
    steps = memory.get_succinct_steps()
    path.write_text(
        json.dumps(steps, ensure_ascii=False, indent=2, default=str)
    )


def load_memory(path: Path) -> list[dict]:
    """Load memory records from a JSON file.

    Returns raw dict list for inspection only; not fed back into
    agent context. Returns an empty list if the file does not exist.
    """
    if not path.is_file():
        return []
    return json.loads(path.read_text())
