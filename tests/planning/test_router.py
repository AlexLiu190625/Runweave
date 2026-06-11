"""Tests for Router model-selection logic."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from runweave.planning import (
    ModelProfile,
    NoCompatibleModelError,
    Router,
    StepMetadata,
)


def _fake_model(model_id: str):
    return SimpleNamespace(model_id=model_id)


def _profile(
    model_id: str,
    *,
    context_window: int = 200_000,
    supports_tools: bool = True,
    supports_structured_output: bool = True,
    coding_score: float = 0.8,
    long_context_score: float = 0.7,
    latency: str = "medium",
    cost_tier: str = "medium",
) -> ModelProfile:
    return ModelProfile(
        model=_fake_model(model_id),
        context_window=context_window,
        supports_tools=supports_tools,
        supports_structured_output=supports_structured_output,
        coding_score=coding_score,
        long_context_score=long_context_score,
        latency=latency,
        cost_tier=cost_tier,
    )


def _metadata(**overrides) -> StepMetadata:
    defaults = dict(
        kind="edit",
        needs_long_context=False,
        needs_structured_output=False,
        needs_tools=False,
        difficulty="medium",
        style="precise",
    )
    defaults.update(overrides)
    return StepMetadata(**defaults)


# ---------------------------------------------------------------------------
# Hard constraints
# ---------------------------------------------------------------------------


def test_hard_constraint_filters_no_tool_support() -> None:
    router = Router()
    p_no_tools = _profile("p1", supports_tools=False)
    p_with_tools = _profile("p2", supports_tools=True)
    chosen = router.select(_metadata(needs_tools=True), [p_no_tools, p_with_tools])
    assert chosen.model_id == "p2"


def test_hard_constraint_filters_no_structured_output() -> None:
    router = Router()
    p_no_struct = _profile("p1", supports_structured_output=False)
    p_with_struct = _profile("p2", supports_structured_output=True)
    chosen = router.select(
        _metadata(needs_structured_output=True), [p_no_struct, p_with_struct]
    )
    assert chosen.model_id == "p2"


def test_hard_constraint_filters_insufficient_context_window() -> None:
    router = Router(long_context_threshold=200_000)
    p_small = _profile("p1", context_window=128_000)
    p_large = _profile("p2", context_window=1_000_000)
    chosen = router.select(_metadata(needs_long_context=True), [p_small, p_large])
    assert chosen.model_id == "p2"


def test_no_compatible_profile_raises() -> None:
    router = Router()
    p1 = _profile("p1", supports_tools=False)
    p2 = _profile("p2", supports_tools=False)
    with pytest.raises(NoCompatibleModelError):
        router.select(_metadata(needs_tools=True), [p1, p2])


# ---------------------------------------------------------------------------
# Soft preferences
# ---------------------------------------------------------------------------


def test_high_difficulty_prefers_high_tier() -> None:
    router = Router()
    low_tier = _profile("p1", cost_tier="low")
    high_tier = _profile("p2", cost_tier="high")
    chosen = router.select(_metadata(difficulty="high"), [low_tier, high_tier])
    assert chosen.model_id == "p2"


def test_low_difficulty_prefers_low_cost() -> None:
    router = Router()
    # At low difficulty, the cost penalty dominates; equal coding_score otherwise.
    low_cost = _profile("p1", cost_tier="low", coding_score=0.7)
    high_cost = _profile("p2", cost_tier="high", coding_score=0.7)
    chosen = router.select(_metadata(difficulty="low"), [low_cost, high_cost])
    assert chosen.model_id == "p1"


def test_edit_kind_prefers_higher_coding_score() -> None:
    router = Router()
    weak = _profile("p1", coding_score=0.5)
    strong = _profile("p2", coding_score=0.95)
    chosen = router.select(_metadata(kind="edit"), [weak, strong])
    assert chosen.model_id == "p2"


def test_long_context_task_prefers_higher_long_context_score() -> None:
    router = Router()
    p_low_lc = _profile("p1", long_context_score=0.4, context_window=1_000_000)
    p_high_lc = _profile("p2", long_context_score=0.9, context_window=1_000_000)
    chosen = router.select(
        _metadata(needs_long_context=True, kind="read"), [p_low_lc, p_high_lc]
    )
    assert chosen.model_id == "p2"


def test_exploratory_style_prefers_low_latency() -> None:
    router = Router()
    # Equal everything else, exploratory should favor low-latency.
    slow = _profile("p1", latency="high", cost_tier="low", coding_score=0.7)
    fast = _profile("p2", latency="low", cost_tier="low", coding_score=0.7)
    chosen = router.select(
        _metadata(style="exploratory", kind="research"), [slow, fast]
    )
    assert chosen.model_id == "p2"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_profiles_raises() -> None:
    router = Router()
    with pytest.raises(NoCompatibleModelError):
        router.select(_metadata(), [])


def test_single_compatible_profile_returned() -> None:
    router = Router()
    only = _profile("only")
    chosen = router.select(_metadata(), [only])
    assert chosen.model_id == "only"


def test_tie_break_returns_first_in_input_order() -> None:
    """When two profiles tie on score, Router returns the first one."""
    router = Router()
    a = _profile("a", cost_tier="medium", coding_score=0.8, long_context_score=0.7)
    b = _profile("b", cost_tier="medium", coding_score=0.8, long_context_score=0.7)
    chosen = router.select(_metadata(), [a, b])
    assert chosen.model_id == "a"
    # And reverse order should pick "b"
    chosen_rev = router.select(_metadata(), [b, a])
    assert chosen_rev.model_id == "b"


# ---------------------------------------------------------------------------
# End-to-end realistic scenarios
# ---------------------------------------------------------------------------


def _frontier_set() -> list[ModelProfile]:
    """A realistic mixed set: small/cheap, mid, top-tier."""
    return [
        _profile(
            "haiku",
            context_window=200_000,
            coding_score=0.7,
            long_context_score=0.65,
            latency="low",
            cost_tier="low",
        ),
        _profile(
            "sonnet",
            context_window=1_000_000,
            coding_score=0.9,
            long_context_score=0.85,
            latency="medium",
            cost_tier="medium",
        ),
        _profile(
            "opus",
            context_window=1_000_000,
            coding_score=0.95,
            long_context_score=0.9,
            latency="high",
            cost_tier="high",
        ),
    ]


def test_scenario_simple_read_picks_cheap() -> None:
    router = Router()
    chosen = router.select(
        _metadata(kind="read", difficulty="low", style="exploratory"),
        _frontier_set(),
    )
    assert chosen.model_id == "haiku"


def test_scenario_hard_edit_picks_top_tier() -> None:
    router = Router()
    chosen = router.select(
        _metadata(kind="edit", difficulty="high", style="precise", needs_tools=True),
        _frontier_set(),
    )
    assert chosen.model_id == "opus"


def test_scenario_long_context_summarize_picks_high_lc_score() -> None:
    router = Router()
    chosen = router.select(
        _metadata(
            kind="summarize",
            difficulty="medium",
            needs_long_context=True,
            style="careful",
        ),
        _frontier_set(),
    )
    # haiku has long_context_score=0.65 but context_window=200_000 which is exactly
    # the threshold; the >= check keeps it. The decisive factor is long_context_score
    # combined with difficulty/tier weighting — opus's higher scores should win.
    assert chosen.model_id == "opus"
