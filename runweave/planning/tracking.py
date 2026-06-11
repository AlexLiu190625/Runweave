from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents.models import Model


@dataclass
class TokenUsageTracker:
    """Accumulator for input/output tokens across multiple Model calls.

    Holds a lock for safety even though PlanningRuntime is single-threaded;
    cheap to maintain and protects future async/parallel callers.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def add(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.call_count += 1

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class TrackedModel:
    """Transparent wrapper around a smolagents Model that accumulates
    token_usage from every call into a shared TokenUsageTracker.

    Quacks like a Model: ``__call__`` matches signature, ``model_id`` is
    exposed, and unknown attribute access forwards to the wrapped instance.
    """

    def __init__(self, wrapped: "Model", tracker: TokenUsageTracker) -> None:
        self._wrapped = wrapped
        self._tracker = tracker

    @property
    def model_id(self) -> str:
        return self._wrapped.model_id

    def __call__(self, messages, **kwargs):
        result = self._wrapped(messages, **kwargs)
        usage = getattr(result, "token_usage", None)
        if usage is not None:
            self._tracker.add(
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0),
            )
        return result

    def __getattr__(self, name: str):
        # Fallback for any attribute the wrapper doesn't explicitly shadow.
        return getattr(self._wrapped, name)
