from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents.models import Model


@dataclass
class ModelProfile:
    """A configured Model instance plus a descriptor of its capabilities.

    coding_score and long_context_score are hand-labeled 0.0-1.0 estimates.
    See README for a recommended baseline table covering common frontier models.
    """

    model: "Model"
    context_window: int
    supports_tools: bool
    supports_structured_output: bool
    coding_score: float
    long_context_score: float
    latency: Literal["low", "medium", "high"]
    cost_tier: Literal["low", "medium", "high"]

    @property
    def model_id(self) -> str:
        return self.model.model_id
