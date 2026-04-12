from __future__ import annotations

from typing import TYPE_CHECKING

from runweave.context.budget import ContextBudget

if TYPE_CHECKING:
    from smolagents.memory import MemoryStep

# Max characters for observations after compression
_MAX_OBS_LEN = 200
# Number of recent steps to keep with full detail
_KEEP_RECENT = 3
# Compression trigger threshold (fraction of step_budget)
_TRIGGER_RATIO = 0.8


class StepCompressor:
    """Intra-run step compressor that reduces context by modifying old step fields.

    Uses three progressive compression tiers (only ActionStep, not TaskStep/SystemPromptStep):
    - Tier 1: truncate observations to _MAX_OBS_LEN characters
    - Tier 2: clear observations, keep code_action
    - Tier 3: observations = "[compressed]", code_action = None
    """

    def __init__(
        self,
        budget: ContextBudget,
        keep_recent: int = _KEEP_RECENT,
    ) -> None:
        self.budget = budget
        self.keep_recent = keep_recent
        self._threshold = int(budget.step_budget() * _TRIGGER_RATIO)

    def compress_if_needed(self, steps: list[MemoryStep]) -> None:
        """Check the latest step's input_tokens and compress old steps if above threshold.

        Uses the most recent ActionStep's token_usage.input_tokens as the actual
        context size. The callback fires before the current step is appended to
        steps, so steps[-1] is the previous step.
        """
        from smolagents.memory import ActionStep

        # Find the most recent ActionStep with token_usage
        last_input_tokens = self._get_last_input_tokens(steps)
        if last_input_tokens is None or last_input_tokens < self._threshold:
            return

        # Collect compressible ActionStep indices (excluding the most recent keep_recent)
        action_indices = [
            i for i, s in enumerate(steps) if isinstance(s, ActionStep)
        ]
        if len(action_indices) <= self.keep_recent:
            return

        compressible = action_indices[: -self.keep_recent]

        # Progressive compression: apply Tier 1 to all, then Tier 2 if needed, then Tier 3
        for tier in range(1, 4):
            for idx in compressible:
                self._apply_tier(steps[idx], tier)
            # Check whether compression is sufficient (rough estimate based on field lengths).
            # Since we can't know the exact next input_tokens, we apply the current tier
            # to all compressible steps at once. The actual effect will be reflected in
            # the next step's token_usage.
            break_after = self._estimate_reduction_sufficient(
                steps, last_input_tokens
            )
            if break_after:
                break

    def _apply_tier(self, step: MemoryStep, tier: int) -> None:
        """Apply the specified compression tier to a single step."""
        from smolagents.memory import ActionStep

        if not isinstance(step, ActionStep):
            return

        if tier == 1:
            # Truncate observations
            if step.observations and len(step.observations) > _MAX_OBS_LEN:
                step.observations = step.observations[:_MAX_OBS_LEN] + "..."
        elif tier == 2:
            # Clear observations, keep code
            step.observations = None
            step.model_output = None
        elif tier == 3:
            # Clear everything
            step.observations = "[compressed]"
            step.code_action = None
            step.model_output = None

    @staticmethod
    def _get_last_input_tokens(steps: list[MemoryStep]) -> int | None:
        """Find the input_tokens of the most recent step with token_usage."""
        from smolagents.memory import ActionStep

        for step in reversed(steps):
            if isinstance(step, ActionStep) and step.token_usage:
                return step.token_usage.input_tokens
        return None

    @staticmethod
    def _estimate_reduction_sufficient(
        steps: list[MemoryStep],
        original_tokens: int,
    ) -> bool:
        """Roughly estimate whether compression is sufficient based on text length changes.

        This is an approximate check: if the total text length has decreased by
        more than 30%, we consider compression sufficient and skip higher tiers.
        """
        from smolagents.memory import ActionStep

        total_chars = 0
        for step in steps:
            if isinstance(step, ActionStep):
                if step.observations:
                    total_chars += len(step.observations)
                if step.code_action:
                    total_chars += len(step.code_action)
                if step.model_output and isinstance(step.model_output, str):
                    total_chars += len(step.model_output)

        # Rough estimate: check if remaining text tokens are below threshold
        estimated_tokens = total_chars / 3.5
        return estimated_tokens < original_tokens * 0.7
