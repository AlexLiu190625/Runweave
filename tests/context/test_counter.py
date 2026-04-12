from __future__ import annotations

from runweave.context.counter import TokenCounter


def test_none_returns_zero():
    assert TokenCounter.estimate(None) == 0


def test_empty_returns_zero():
    assert TokenCounter.estimate("") == 0


def test_short_text():
    # "hello" = 5 chars / 3.5 ≈ 1.43 → ceil = 2
    assert TokenCounter.estimate("hello") == 2


def test_longer_text():
    text = "x" * 100
    result = TokenCounter.estimate(text)
    # 100 / 3.5 ≈ 28.57 → ceil = 29
    assert result == 29


def test_proportional():
    """Longer text should produce proportionally more tokens."""
    short = TokenCounter.estimate("x" * 100)
    long = TokenCounter.estimate("x" * 1000)
    assert long > short
    # ceil rounding causes small deviations at scale
    assert abs(long - short * 10) <= 5
