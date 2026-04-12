from __future__ import annotations

from dataclasses import dataclass

# 模型 ID（子串匹配）→ context window 大小（tokens）
# 匹配时从最长 key 开始，避免短 key 误匹配
# 只收录当前前沿模型，过时模型走 _DEFAULT_WINDOW
# 最后更新：2026-04
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI
    "gpt-5.4-mini": 1_050_000,
    "gpt-5.4": 1_050_000,
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
    """从 MODEL_CONTEXT_WINDOWS 查表，子串匹配，最长 key 优先。"""
    model_lower = model_id.lower()
    # 按 key 长度降序匹配，防止短 key 误匹配
    for key in sorted(MODEL_CONTEXT_WINDOWS, key=len, reverse=True):
        if key in model_lower:
            return MODEL_CONTEXT_WINDOWS[key]
    return _DEFAULT_WINDOW


@dataclass
class ContextBudget:
    """根据模型 context window 计算 token 预算分配。"""

    model_id: str
    buffer_tokens: int = 4096
    instruction_ratio: float = 0.25

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
