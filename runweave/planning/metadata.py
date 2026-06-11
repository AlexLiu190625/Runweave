from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StepKind = Literal["read", "edit", "test", "summarize", "research"]
Difficulty = Literal["low", "medium", "high"]
Style = Literal["careful", "exploratory", "precise"]


@dataclass(frozen=True)
class StepMetadata:
    """Describes what a planning step needs from its model.

    Six frozen fields drive the Router's hard constraints and soft preferences.
    Frozen so it can be safely used as a dict key and to prevent runtime mutation.
    """

    kind: StepKind
    needs_long_context: bool
    needs_structured_output: bool
    needs_tools: bool
    difficulty: Difficulty
    style: Style
