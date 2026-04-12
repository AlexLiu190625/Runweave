from __future__ import annotations

from runweave.context.budget import ContextBudget, _DEFAULT_WINDOW


def test_known_model_exact():
    b = ContextBudget("gpt-4.1")
    assert b.context_window == 1_047_576


def test_known_model_substring():
    """model_id 包含已知 key 时应匹配。"""
    b = ContextBudget("claude-sonnet-4-20250514")
    assert b.context_window == 200_000


def test_longest_key_wins():
    """gpt-4.1-mini 应匹配 'gpt-4.1-mini' 而非 'gpt-4.1'。"""
    b = ContextBudget("gpt-4.1-mini-2025-04-14")
    assert b.context_window == 1_047_576

    b2 = ContextBudget("o4-mini")
    assert b2.context_window == 200_000


def test_frontier_models():
    """验证 2026 前沿模型的窗口大小。"""
    assert ContextBudget("gpt-5.4").context_window == 1_050_000
    assert ContextBudget("claude-opus-4-6").context_window == 1_000_000
    assert ContextBudget("claude-sonnet-4-6").context_window == 1_000_000
    assert ContextBudget("gemini-2.5-pro").context_window == 1_048_576
    assert ContextBudget("deepseek-chat").context_window == 163_840
    assert ContextBudget("llama-4-scout").context_window == 10_000_000
    assert ContextBudget("qwen3-max").context_window == 1_000_000
    assert ContextBudget("glm-5").context_window == 200_000


def test_unknown_model_fallback():
    b = ContextBudget("some-random-model")
    assert b.context_window == _DEFAULT_WINDOW


def test_budget_partition():
    b = ContextBudget("gpt-4.1", buffer_tokens=4096, instruction_ratio=0.25)
    assert b.available == 1_047_576 - 4096
    assert b.instruction_budget() == int(b.available * 0.25)
    assert b.step_budget() == b.available - b.instruction_budget()


def test_budget_math_identity():
    """instruction_budget + step_budget == available。"""
    b = ContextBudget("gpt-4.1")
    assert b.instruction_budget() + b.step_budget() == b.available


def test_custom_ratios():
    b = ContextBudget("gpt-4.1", instruction_ratio=0.4)
    assert b.instruction_budget() == int(b.available * 0.4)
