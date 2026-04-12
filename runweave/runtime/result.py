from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunResult:
    """Return value of Runtime.run()."""

    # Final output from the agent
    output: Any
    # ID of the owning thread
    thread_id: str
    # Execution state: "success" or "max_steps_error"
    state: str
    # Number of steps executed in this run
    step_count: int
    # Token usage statistics (may be None)
    token_usage: dict | None
    # Timing statistics (may be None)
    timing: dict | None
    # Thread summary
    summary: str | None = None
    # Skills loaded during this run
    skills_used: list[str] = field(default_factory=list)
