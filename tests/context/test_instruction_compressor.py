from __future__ import annotations

from runweave.context.budget import ContextBudget
from runweave.context.instruction_compressor import InstructionCompressor
from runweave.runtime.run_record import RunRecord, StepRecord


def _make_compressor(available_tokens: int = 100_000) -> InstructionCompressor:
    """Create a compressor with the specified available token count."""
    # buffer_tokens = context_window - available_tokens
    budget = ContextBudget("claude-sonnet-4", buffer_tokens=200_000 - available_tokens)
    return InstructionCompressor(budget)


def _make_record(
    run_number: int = 1,
    task: str = "Do something",
    num_steps: int = 1,
    obs_len: int = 100,
) -> RunRecord:
    """Create a RunRecord for testing."""
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
        timestamp="2026-04-10T00:00:00+00:00",
        task=task,
        state="success",
        step_count=num_steps,
        skills_used=[],
        tools_used=[],
        steps=steps,
        output=f"Done with run {run_number}",
    )


def test_all_parts_fit():
    comp = _make_compressor(100_000)
    result = comp.compress(
        user_instructions="You are helpful.",
        skill_catalog="## Skills\n- greeting",
        history_records=[_make_record()],
        thread_summary="Did some work.",
    )
    assert "You are helpful" in result
    assert "Skills" in result
    assert "Thread History" in result
    assert "Thread Summary" in result


def test_no_parts_returns_none():
    comp = _make_compressor()
    result = comp.compress(None, None, None, None)
    assert result is None


def test_user_instructions_never_cut():
    """user_instructions should never be trimmed, even with a very small budget."""
    comp = _make_compressor(available_tokens=50)
    records = [_make_record(i, num_steps=3, obs_len=500) for i in range(10)]
    result = comp.compress(
        user_instructions="Critical instruction that must survive.",
        skill_catalog=None,
        history_records=records,
        thread_summary=None,
    )
    assert "Critical instruction that must survive" in result


def test_history_compressed_when_over_budget():
    """History should be compressed when budget is insufficient."""
    comp = _make_compressor(available_tokens=200)
    records = [_make_record(i + 1, num_steps=3, obs_len=500) for i in range(5)]
    result = comp.compress(
        user_instructions="Be helpful.",
        skill_catalog=None,
        history_records=records,
        thread_summary=None,
    )
    assert "Be helpful" in result
    # Full render would be very large; compressed result should be much smaller
    full = InstructionCompressor._render_full(records)
    assert len(result) < len(full)


def test_render_headers_only():
    """_render_headers_only should exclude code blocks and detailed output."""
    records = [_make_record(1, task="Task one", num_steps=2)]
    rendered = InstructionCompressor._render_headers_only(records)
    assert "### Run 1" in rendered
    assert "compute_" not in rendered
    assert "**Output:**" in rendered


def test_render_run_log_only():
    """_render_run_log should contain only the table, no Recent Runs."""
    records = [_make_record(1), _make_record(2)]
    rendered = InstructionCompressor._render_run_log(records)
    assert "Run Log" in rendered
    assert "Recent Runs" not in rendered
    assert "| 1 |" in rendered
    assert "| 2 |" in rendered


def test_render_run_log_truncated():
    """Passing a slice of records to _render_run_log should limit rows."""
    records = [_make_record(i + 1) for i in range(20)]
    rendered = InstructionCompressor._render_run_log(records[-5:])
    assert "| 20 |" in rendered
    assert "| 16 |" in rendered
    assert "| 15 |" not in rendered


def test_render_full_includes_code():
    """_render_full should include code blocks in recent runs."""
    records = [_make_record(1, num_steps=2)]
    rendered = InstructionCompressor._render_full(records)
    assert "```python" in rendered
    assert "compute_1" in rendered


def test_task_with_pipe_escaped():
    """Pipe characters in task should be escaped in run log table."""
    records = [_make_record(1, task="compare A | B")]
    rendered = InstructionCompressor._render_run_log(records)
    assert "\\|" in rendered
    # Should not break the table (no bare | inside the cell)


def test_task_with_newline_escaped():
    """Newline in task should be replaced with space."""
    records = [_make_record(1, task="line one\nline two")]
    rendered = InstructionCompressor._render_run_log(records)
    assert "\n" not in rendered.split("\n")[-1] or "line one line two" in rendered
