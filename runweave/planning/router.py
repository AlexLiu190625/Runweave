from __future__ import annotations

from runweave.planning.errors import NoCompatibleModelError
from runweave.planning.metadata import StepMetadata
from runweave.planning.profile import ModelProfile

# Soft-preference scoring weights.
# Heuristic; revisit after collecting real usage telemetry.
_DIFFICULTY_WEIGHT = {"low": 1.0, "medium": 1.5, "high": 2.0}
_TIER_SCORE = {"low": 0.3, "medium": 0.6, "high": 1.0}
_LATENCY_BONUS = {"low": 0.5, "medium": 0.2, "high": 0.0}
_COST_PENALTY = {"low": 0.0, "medium": -0.5, "high": -1.0}

_DEFAULT_LONG_CONTEXT_THRESHOLD = 200_000


class Router:
    """Select a ModelProfile for a step based on its StepMetadata.

    Two-phase selection:
      1. Hard constraints filter (must-haves) - profiles that fail are discarded
      2. Soft preference scoring - the surviving profile with the highest score wins

    Tie-break is deterministic: max() returns the first maximum in input order.
    """

    def __init__(self, long_context_threshold: int | None = None) -> None:
        self.long_context_threshold = (
            long_context_threshold
            if long_context_threshold is not None
            else _DEFAULT_LONG_CONTEXT_THRESHOLD
        )

    def select(
        self,
        metadata: StepMetadata,
        profiles: list[ModelProfile],
    ) -> ModelProfile:
        candidates = [p for p in profiles if self._passes_hard_constraints(p, metadata)]
        if not candidates:
            raise NoCompatibleModelError(metadata, len(profiles))
        return max(candidates, key=lambda p: self._score(p, metadata))

    def _passes_hard_constraints(
        self, profile: ModelProfile, metadata: StepMetadata
    ) -> bool:
        if metadata.needs_tools and not profile.supports_tools:
            return False
        if metadata.needs_structured_output and not profile.supports_structured_output:
            return False
        if (
            metadata.needs_long_context
            and profile.context_window < self.long_context_threshold
        ):
            return False
        return True

    def _score(self, profile: ModelProfile, metadata: StepMetadata) -> float:
        s = 0.0
        if metadata.kind in {"edit", "test"}:
            s += 2.0 * profile.coding_score
        if metadata.needs_long_context:
            s += 2.0 * profile.long_context_score
        s += _DIFFICULTY_WEIGHT[metadata.difficulty] * _TIER_SCORE[profile.cost_tier]
        if metadata.style in {"precise", "careful"}:
            s += profile.coding_score
        elif metadata.style == "exploratory":
            s += _LATENCY_BONUS[profile.latency]
        s += _COST_PENALTY[profile.cost_tier]
        return s
