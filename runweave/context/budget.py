from __future__ import annotations

from dataclasses import dataclass

# Model ID (substring match) -> context window size (tokens)
# Matches longest key first to avoid short-key false matches
# Only includes current frontier models; older models use _DEFAULT_WINDOW
# Last updated: 2026-04
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI
    "gpt-5.4-mini": 1_050_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.3": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1": 1_047_576,
    "o4-mini": 200_000,
    "o3": 200_000,
    # Anthropic
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4": 200_000,
    # Google
    "gemini-3.1": 1_000_000,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    # DeepSeek
    "deepseek-chat": 163_840,
    "deepseek-reasoner": 163_840,
    # Meta
    "llama-4-scout": 10_000_000,
    "llama-4-maverick": 1_048_576,
    # Mistral
    "mistral-medium-3": 131_072,
    "mistral-large": 256_000,
    # Qwen
    "qwen3": 1_000_000,
    "qwen-2.5": 131_072,
    # GLM
    "glm-5": 200_000,
}

_DEFAULT_WINDOW = 32_000


def _lookup_context_window(model_id: str) -> int:
    """Look up context window from MODEL_CONTEXT_WINDOWS via substring match, longest key first."""
    model_lower = model_id.lower()
    # Sort by key length descending to prevent short-key false matches
    for key in sorted(MODEL_CONTEXT_WINDOWS, key=len, reverse=True):
        if key in model_lower:
            return MODEL_CONTEXT_WINDOWS[key]
    return _DEFAULT_WINDOW


@dataclass
class ContextBudget:
    """Compute token budget allocation based on model context window.

    ``head_count`` and ``tail_count`` control U-shaped history decay in
    ``InstructionCompressor``: the first ``head_count`` runs and the last
    ``tail_count`` runs are always rendered FULL; middle runs are progressively
    compressed when budget is tight.

    For very long threads (N > 20) using both ``key_facts`` and
    ``thread_summary``, ``head_count=0`` is a reasonable choice — the early
    runs' direction information is already captured by those tracks, and
    pinning them FULL wastes budget that could go to middle runs.
    """

    model_id: str
    buffer_tokens: int = 4096
    instruction_ratio: float = 0.25
    head_count: int = 2
    tail_count: int = 3

    def __post_init__(self) -> None:
        if self.head_count < 0:
            raise ValueError("head_count must be >= 0")
        if self.tail_count < 1:
            raise ValueError("tail_count must be >= 1")

    @property
    def context_window(self) -> int:
        return _lookup_context_window(self.model_id)

    @property
    def available(self) -> int:
        return self.context_window - self.buffer_tokens

    def instruction_budget(self) -> int:
        return int(self.available * self.instruction_ratio)

    def step_budget(self) -> int:
        return self.available - self.instruction_budget()
