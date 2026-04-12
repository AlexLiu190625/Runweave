from __future__ import annotations

from typing import Any, Callable

from runweave.context.budget import ContextBudget
from runweave.context.step_compressor import StepCompressor


def make_context_callback(budget: ContextBudget) -> Callable:
    """Create a smolagents step callback that compresses memory when above threshold.

    The returned callback is registered via CodeAgent(step_callbacks={ActionStep: callback})
    and fires after each step completes. The callback executes before the step is
    appended to memory, so compression takes effect immediately for the next step.
    """
    compressor = StepCompressor(budget)

    def _callback(step: Any, **kwargs: Any) -> None:
        agent = kwargs.get("agent")
        if agent is None:
            return
        compressor.compress_if_needed(agent.memory.steps)

    return _callback
