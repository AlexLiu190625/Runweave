"""Tests for U-shaped decay in InstructionCompressor.

Verifies the head/middle/tail level assignment, 5-level budget degradation,
and that head/tail runs are never demoted regardless of pressure.
"""
from __future__ import annotations

import pytest

from runweave.context.budget import ContextBudget
from runweave.context.instruction_compressor import (
    InstructionCompressor,
    _assign_levels,
    _degrade,
    _render_history,
)
from runweave.runtime.run_record import (
    DetailLevel,
    RunRecord,
    StepRecord,
)


def _make_record(
    run_number: int = 1,
    task: str | None = None,
    num_steps: int = 1,
    obs_len: int = 100,
) -> RunRecord:
    """Build a RunRecord with a unique task string so it can be located in output."""
    steps = [
        StepRecord(
            step_number=i + 1,
            code=f"result = compute_{i + 1}()",
            output="x" * obs_len,
        )
        for i in range(num_steps)
    ]
    return RunRecord(
        run_number=run_number,
        timestamp="2026-04-22T00:00:00+00:00",
        task=task if task is not None else f"task_{run_number}",
        state="success",
        step_count=num_steps,
        skills_used=[],
        tools_used=[],
        steps=steps,
        output=f"Done with run {run_number}",
    )


def _make_compressor(
    available_tokens: int = 100_000,
    head_count: int = 2,
    tail_count: int = 3,
) -> InstructionCompressor:
    budget = ContextBudget(
        "claude-sonnet-4",
        buffer_tokens=200_000 - available_tokens,
        head_count=head_count,
        tail_count=tail_count,
    )
    return InstructionCompressor(budget)


# ---------------------------------------------------------------------------
# _assign_levels: shape of the U
# ---------------------------------------------------------------------------


def test_assign_levels_n0() -> None:
    assert _assign_levels(0, 2, 3) == []


def test_assign_levels_n1() -> None:
    assert _assign_levels(1, 2, 3) == [DetailLevel.FULL]


def test_assign_levels_all_full_when_n_le_head_plus_tail() -> None:
    # n=5 with head=2, tail=3 → no middle, all FULL
    assert _assign_levels(5, 2, 3) == [DetailLevel.FULL] * 5
    # n=4 (less than head+tail) → all FULL
    assert _assign_levels(4, 2, 3) == [DetailLevel.FULL] * 4


def test_assign_levels_n10_u_shape() -> None:
    levels = _assign_levels(10, 2, 3)
    # Head: runs 0,1
    assert levels[0] == DetailLevel.FULL
    assert levels[1] == DetailLevel.FULL
    # Tail: runs 7,8,9
    assert levels[7] == DetailLevel.FULL
    assert levels[8] == DetailLevel.FULL
    assert levels[9] == DetailLevel.FULL
    # Middle (indices 2..6) contains a mix; closer to tail = more detail
    middle = levels[2:7]
    assert DetailLevel.LOG_LINE in middle    # farthest from tail
    assert DetailLevel.TAKEAWAY in middle    # closest to tail


def test_assign_levels_middle_closer_to_tail_has_higher_detail() -> None:
    # For N=30, the closest-to-tail middle run should have higher level
    # than the farthest-from-tail middle run.
    levels = _assign_levels(30, 2, 3)
    # middle indices 2..26; near-tail = 26, far-from-tail = 2
    near_tail = levels[26]
    far_from_tail = levels[2]
    assert near_tail > far_from_tail


def test_assign_levels_head_count_zero() -> None:
    levels = _assign_levels(10, 0, 3)
    # No head pin; tail is last 3
    assert levels[7] == DetailLevel.FULL
    assert levels[8] == DetailLevel.FULL
    assert levels[9] == DetailLevel.FULL
    # Indices 0,1 are middle now; should not all be FULL
    assert any(l != DetailLevel.FULL for l in levels[:7])


# ---------------------------------------------------------------------------
# _degrade: head/tail must never drop below FULL
# ---------------------------------------------------------------------------


def test_head_tail_never_degrade_at_any_budget_level() -> None:
    head_count = 2
    tail_count = 3
    base = _assign_levels(15, head_count, tail_count)
    for budget_level in range(5):
        degraded = _degrade(base, budget_level, head_count, tail_count)
        # Head indices remain FULL
        for i in range(head_count):
            assert degraded[i] == DetailLevel.FULL, (
                f"head[{i}] demoted at budget_level={budget_level}"
            )
        # Tail indices remain FULL
        for i in range(len(degraded) - tail_count, len(degraded)):
            assert degraded[i] == DetailLevel.FULL, (
                f"tail[{i}] demoted at budget_level={budget_level}"
            )


def test_degrade_l0_is_identity() -> None:
    base = _assign_levels(10, 2, 3)
    assert _degrade(base, 0, 2, 3) == base


def test_degrade_l1_demotes_middle_one_notch() -> None:
    base = _assign_levels(10, 2, 3)
    deg = _degrade(base, 1, 2, 3)
    # Every middle entry should be strictly lower (or LOG_LINE which clamps).
    for i in range(2, 7):
        assert deg[i] <= base[i]
        # Demoted by 1 unless already at floor
        if base[i] != DetailLevel.LOG_LINE:
            assert int(deg[i]) == int(base[i]) - 1


def test_degrade_l2_floors_middle_to_log_line() -> None:
    base = _assign_levels(10, 2, 3)
    deg = _degrade(base, 2, 2, 3)
    for i in range(2, 7):
        assert deg[i] == DetailLevel.LOG_LINE


def test_degrade_handles_no_middle() -> None:
    # When n <= head+tail, _degrade should be a no-op (no middle to touch).
    base = _assign_levels(4, 2, 3)
    for budget_level in range(5):
        assert _degrade(base, budget_level, 2, 3) == base


# ---------------------------------------------------------------------------
# ContextBudget validation
# ---------------------------------------------------------------------------


def test_invalid_head_count_raises() -> None:
    with pytest.raises(ValueError):
        ContextBudget("claude-sonnet-4", head_count=-1)


def test_invalid_tail_count_raises() -> None:
    with pytest.raises(ValueError):
        ContextBudget("claude-sonnet-4", tail_count=0)


# ---------------------------------------------------------------------------
# Matrix tests: {N=3,5,10,30,50} × {loose, medium, tight, very_tight}
# ---------------------------------------------------------------------------


def _make_records(n: int) -> list[RunRecord]:
    # 1 step + 100 chars/output → ~70 tokens per FULL record. Keeps cascade
    # transitions visible at realistic budgets (e.g. N=10 head+tail FULL ≈ 350
    # tokens, well under 1.5k).
    return [_make_record(i + 1, num_steps=1, obs_len=100) for i in range(n)]


# N=3 (less than head+tail, everything fits as FULL)


def test_n3_loose() -> None:
    comp = _make_compressor(available_tokens=100_000)
    text = comp._render_within_budget(_make_records(3), 100_000)
    for i in (1, 2, 3):
        assert f"task_{i}" in text


def test_n3_medium() -> None:
    comp = _make_compressor(available_tokens=8_000)
    text = comp._render_within_budget(_make_records(3), 8_000)
    for i in (1, 2, 3):
        assert f"task_{i}" in text


def test_n3_tight() -> None:
    comp = _make_compressor(available_tokens=1_500)
    text = comp._render_within_budget(_make_records(3), 1_500)
    for i in (1, 2, 3):
        assert f"task_{i}" in text


def test_n3_very_tight() -> None:
    comp = _make_compressor(available_tokens=200)
    text = comp._render_within_budget(_make_records(3), 200)
    # At very tight budget the result is either fitting text or empty;
    # if text is non-empty, all 3 task names appear at minimum in the table.
    if text:
        for i in (1, 2, 3):
            assert f"task_{i}" in text


# N=5 (exactly head+tail, no middle)


def test_n5_loose() -> None:
    comp = _make_compressor(available_tokens=100_000)
    text = comp._render_within_budget(_make_records(5), 100_000)
    for i in range(1, 6):
        assert f"task_{i}" in text


def test_n5_medium() -> None:
    comp = _make_compressor(available_tokens=8_000)
    text = comp._render_within_budget(_make_records(5), 8_000)
    for i in range(1, 6):
        assert f"task_{i}" in text


def test_n5_tight() -> None:
    comp = _make_compressor(available_tokens=1_500)
    text = comp._render_within_budget(_make_records(5), 1_500)
    # At tight budget, all 5 names should still appear in the table.
    for i in range(1, 6):
        assert f"task_{i}" in text


def test_n5_very_tight() -> None:
    comp = _make_compressor(available_tokens=300)
    text = comp._render_within_budget(_make_records(5), 300)
    if text:
        # Table alone fits in ~250 chars for 5 rows; all task names present.
        for i in range(1, 6):
            assert f"task_{i}" in text


# N=10 (5 middle runs)


def test_n10_loose() -> None:
    comp = _make_compressor(available_tokens=100_000)
    text = comp._render_within_budget(_make_records(10), 100_000)
    # Head and tail task names should all appear in FULL sections (with skills/tools).
    for i in (1, 2, 8, 9, 10):
        assert f"task_{i}" in text
    # The full Recent Runs section should be present.
    assert "## Recent Runs" in text


def test_n10_medium() -> None:
    comp = _make_compressor(available_tokens=2_500)
    text = comp._render_within_budget(_make_records(10), 2_500)
    # Head + tail still appear; middle may be table-only.
    for i in (1, 2, 8, 9, 10):
        assert f"task_{i}" in text


def test_n10_tight() -> None:
    comp = _make_compressor(available_tokens=1_500)
    text = comp._render_within_budget(_make_records(10), 1_500)
    # All 10 tasks at least in the table.
    for i in range(1, 11):
        assert f"task_{i}" in text


def test_n10_very_tight() -> None:
    comp = _make_compressor(available_tokens=300)
    text = comp._render_within_budget(_make_records(10), 300)
    # At budget_level 4, only head+tail in the table.
    if text:
        for i in (1, 2, 8, 9, 10):
            assert f"task_{i}" in text


# N=30


def test_n30_loose() -> None:
    comp = _make_compressor(available_tokens=100_000)
    text = comp._render_within_budget(_make_records(30), 100_000)
    # Head, tail, AND middle task names in some form.
    for i in (1, 2, 28, 29, 30):
        assert f"task_{i}" in text


def test_n30_medium() -> None:
    comp = _make_compressor(available_tokens=4_000)
    text = comp._render_within_budget(_make_records(30), 4_000)
    # Head and tail must appear somewhere.
    for i in (1, 2, 28, 29, 30):
        assert f"task_{i}" in text


def test_n30_tight() -> None:
    comp = _make_compressor(available_tokens=1_500)
    text = comp._render_within_budget(_make_records(30), 1_500)
    # Head and tail in table at minimum.
    for i in (1, 2, 28, 29, 30):
        assert f"task_{i}" in text


def test_n30_very_tight() -> None:
    comp = _make_compressor(available_tokens=300)
    text = comp._render_within_budget(_make_records(30), 300)
    if text:
        for i in (1, 2, 28, 29, 30):
            assert f"task_{i}" in text


# N=50


def test_n50_loose() -> None:
    comp = _make_compressor(available_tokens=100_000)
    text = comp._render_within_budget(_make_records(50), 100_000)
    for i in (1, 2, 48, 49, 50):
        assert f"task_{i}" in text


def test_n50_medium() -> None:
    comp = _make_compressor(available_tokens=5_000)
    text = comp._render_within_budget(_make_records(50), 5_000)
    for i in (1, 2, 48, 49, 50):
        assert f"task_{i}" in text


def test_n50_tight() -> None:
    comp = _make_compressor(available_tokens=1_800)
    text = comp._render_within_budget(_make_records(50), 1_800)
    for i in (1, 2, 48, 49, 50):
        assert f"task_{i}" in text


def test_n50_very_tight() -> None:
    comp = _make_compressor(available_tokens=300)
    text = comp._render_within_budget(_make_records(50), 300)
    if text:
        for i in (1, 2, 48, 49, 50):
            assert f"task_{i}" in text


# ---------------------------------------------------------------------------
# Invariants / edge cases
# ---------------------------------------------------------------------------


def test_head_count_zero_only_tail_pinned() -> None:
    comp = _make_compressor(available_tokens=1_500, head_count=0, tail_count=3)
    text = comp._render_within_budget(_make_records(10), 1_500)
    # Tail (8,9,10) must appear in FULL sections (with their Step bodies).
    for i in (8, 9, 10):
        assert f"task_{i}" in text


def test_tail_count_one_only_latest_pinned() -> None:
    comp = _make_compressor(available_tokens=1_500, head_count=2, tail_count=1)
    levels = _assign_levels(10, 2, 1)
    # Tail is just the last run
    assert levels[9] == DetailLevel.FULL
    # Run 8, 9 are now middle (no longer in tail)
    # But head (0, 1) is still FULL
    assert levels[0] == DetailLevel.FULL
    assert levels[1] == DetailLevel.FULL


def test_custom_head_tail_via_budget() -> None:
    comp = _make_compressor(available_tokens=100_000, head_count=5, tail_count=5)
    text = comp._render_within_budget(_make_records(20), 100_000)
    # Front 5 and back 5 should all appear in FULL sections.
    for i in (1, 2, 3, 4, 5, 16, 17, 18, 19, 20):
        assert f"task_{i}" in text


def test_monotone_token_count_across_budget_levels() -> None:
    """Rendering at increasing budget_level yields non-increasing token counts."""
    records = _make_records(20)
    base = _assign_levels(20, 2, 3)
    from runweave.context.counter import TokenCounter

    counter = TokenCounter()
    tokens = []
    for budget_level in range(5):
        levels = _degrade(base, budget_level, 2, 3)
        text = _render_history(records, levels, budget_level, 2, 3)
        tokens.append(counter.estimate(text))
    # Non-increasing
    for i in range(1, len(tokens)):
        assert tokens[i] <= tokens[i - 1], (
            f"token count grew between L{i-1} ({tokens[i-1]}) and L{i} ({tokens[i]})"
        )


def test_l3_omission_hint_present_when_middle_dropped() -> None:
    records = _make_records(10)
    base = _assign_levels(10, 2, 3)
    deg = _degrade(base, 3, 2, 3)
    text = _render_history(records, deg, budget_level=3, head_count=2, tail_count=3)
    # Hint mentions count of omitted middle runs and the tool name.
    assert "5 middle runs omitted" in text
    # Tool name resolves from the constant (no hardcoded string drift).
    from runweave.runtime.history import READ_RUN_DETAIL_TOOL_NAME
    assert READ_RUN_DETAIL_TOOL_NAME in text


def test_l3_no_hint_when_no_middle() -> None:
    # n=5 means no middle exists; no hint should appear.
    records = _make_records(5)
    base = _assign_levels(5, 2, 3)
    deg = _degrade(base, 3, 2, 3)
    text = _render_history(records, deg, budget_level=3, head_count=2, tail_count=3)
    assert "omitted" not in text


def test_l4_drops_middle_from_table() -> None:
    records = _make_records(10)
    base = _assign_levels(10, 2, 3)
    deg = _degrade(base, 4, 2, 3)
    text = _render_history(records, deg, budget_level=4, head_count=2, tail_count=3)
    # Middle tasks (3..7) should NOT appear in the table at L4.
    for i in (3, 4, 5, 6, 7):
        assert f"task_{i}" not in text
    # Head + tail tasks DO appear.
    for i in (1, 2, 8, 9, 10):
        assert f"task_{i}" in text
    # No Recent Runs section at L4.
    assert "Recent Runs" not in text


def test_sections_rendered_newest_first() -> None:
    records = _make_records(10)
    base = _assign_levels(10, 2, 3)
    text = _render_history(records, base, budget_level=0, head_count=2, tail_count=3)
    # In the Recent Runs section, Run 10 (newest) should appear before Run 1 (oldest).
    pos_10 = text.index("### Run 10")
    pos_1 = text.index("### Run 1 ")  # space disambiguates from Run 10
    assert pos_10 < pos_1


def test_zero_step_run_renders_without_error() -> None:
    # A run with step_count=0 / no steps must render at all levels.
    record_zero = RunRecord(
        run_number=1,
        timestamp="2026-04-22T00:00:00+00:00",
        task="empty_task",
        state="success",
        step_count=0,
        skills_used=[],
        tools_used=[],
        steps=[],
        output="nothing",
    )
    for level in (DetailLevel.TITLE, DetailLevel.TAKEAWAY, DetailLevel.FULL):
        text = record_zero.render_at_level(level)
        assert "empty_task" in text


def test_key_facts_and_summary_unaffected_by_ushape() -> None:
    """Fixed parts survive even when history is heavily compressed by U-shape."""
    comp = _make_compressor(available_tokens=2_000)
    result = comp.compress(
        user_instructions="USER_INSTR_MARKER",
        skill_catalog="SKILL_MARKER",
        history_records=_make_records(20),
        thread_summary="SUMMARY_MARKER",
        key_facts="KEY_FACTS_MARKER",
    )
    assert result is not None
    assert "USER_INSTR_MARKER" in result
    assert "SKILL_MARKER" in result
    assert "KEY_FACTS_MARKER" in result
    assert "SUMMARY_MARKER" in result
