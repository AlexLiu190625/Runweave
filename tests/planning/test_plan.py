"""Tests for Plan / PlanStep dataclass and JSON round-trip."""
from __future__ import annotations

import pytest

from runweave.planning import (
    Plan,
    PlanStep,
    StepMetadata,
    UnsupportedPlanVersionError,
)


def _metadata(**overrides) -> StepMetadata:
    defaults = dict(
        kind="edit",
        needs_long_context=False,
        needs_structured_output=False,
        needs_tools=True,
        difficulty="medium",
        style="precise",
    )
    defaults.update(overrides)
    return StepMetadata(**defaults)


def _step(step_id: str, depends_on: list[str] | None = None) -> PlanStep:
    return PlanStep(
        id=step_id,
        title=f"Title {step_id}",
        description=f"Do {step_id}",
        metadata=_metadata(),
        depends_on=depends_on or [],
    )


def test_plan_json_roundtrip() -> None:
    plan = Plan.new(
        task="build CLI",
        steps=[
            _step("s1"),
            _step("s2", depends_on=["s1"]),
        ],
        plan_number=3,
    )
    text = plan.to_json()
    restored = Plan.from_json(text)
    assert restored.task == "build CLI"
    assert restored.plan_number == 3
    assert len(restored.steps) == 2
    assert restored.steps[0].id == "s1"
    assert restored.steps[1].depends_on == ["s1"]


def test_unsupported_version_raises() -> None:
    bad = '{"version": 99, "plan_number": 1, "task": "x", "created_at": "t", "steps": []}'
    with pytest.raises(UnsupportedPlanVersionError):
        Plan.from_json(bad)


def test_next_executable_picks_pending_with_done_deps() -> None:
    plan = Plan.new(
        task="t",
        steps=[
            _step("s1"),
            _step("s2", depends_on=["s1"]),
            _step("s3", depends_on=["s2"]),
        ],
    )
    # Initially s1 is the only one with no unmet deps
    assert plan.next_executable().id == "s1"
    plan.mark_done("s1", "ok")
    assert plan.next_executable().id == "s2"
    plan.mark_done("s2", "ok")
    assert plan.next_executable().id == "s3"
    plan.mark_done("s3", "ok")
    assert plan.next_executable() is None


def test_next_executable_returns_none_when_all_blocked() -> None:
    plan = Plan.new(
        task="t",
        steps=[
            _step("s1"),
            _step("s2", depends_on=["s1"]),
        ],
    )
    plan.mark_failed("s1", "boom")
    plan.mark_blocked_downstream("s1")
    assert plan.next_executable() is None
    assert plan.steps[1].status == "blocked"


def test_mark_failed_blocks_downstream() -> None:
    plan = Plan.new(
        task="t",
        steps=[
            _step("s1"),
            _step("s2", depends_on=["s1"]),
            _step("s3", depends_on=["s2"]),
            _step("s4"),  # independent
        ],
    )
    plan.mark_failed("s1", "fail")
    plan.mark_blocked_downstream("s1")
    assert plan.steps[1].status == "blocked"
    # s3 chain through s2 should also become blocked
    assert plan.steps[2].status == "blocked"
    # s4 is independent — still pending
    assert plan.steps[3].status == "pending"


def test_mark_blocked_downstream_chain_3_levels() -> None:
    """s1 → s2 → s3 chain: failing s1 must block both s2 and s3."""
    plan = Plan.new(
        task="t",
        steps=[
            _step("s1"),
            _step("s2", depends_on=["s1"]),
            _step("s3", depends_on=["s2"]),
        ],
    )
    plan.mark_failed("s1", "boom")
    plan.mark_blocked_downstream("s1")
    assert plan.steps[1].status == "blocked"
    assert plan.steps[2].status == "blocked"


def test_mark_blocked_downstream_out_of_order() -> None:
    """Cascade must succeed even when steps are declared in reverse-dependency
    order — the previous single-pass implementation missed downstream blocking
    in this case (fixpoint loop fixes it).
    """
    # Declaration order: s3 (depends on s2), s2 (depends on s1), s1
    plan = Plan.new(
        task="t",
        steps=[
            _step("s3", depends_on=["s2"]),
            _step("s2", depends_on=["s1"]),
            _step("s1"),
        ],
    )
    plan.mark_failed("s1", "boom")
    plan.mark_blocked_downstream("s1")
    # All three downstream steps should be blocked despite the reversed order
    by_id = {s.id: s for s in plan.steps}
    assert by_id["s2"].status == "blocked"
    assert by_id["s3"].status == "blocked"


def test_is_terminal_when_all_done_or_failed_or_blocked() -> None:
    plan = Plan.new(
        task="t",
        steps=[_step("s1"), _step("s2", depends_on=["s1"])],
    )
    assert not plan.is_terminal()
    plan.mark_failed("s1", "fail")
    plan.mark_blocked_downstream("s1")
    assert plan.is_terminal()
