from __future__ import annotations

from runweave.context.budget import ContextBudget
from runweave.context.instruction_compressor import InstructionCompressor


def _make_compressor(available_tokens: int = 100_000) -> InstructionCompressor:
    """Create a compressor with the specified available token count."""
    # buffer_tokens = context_window - available_tokens
    budget = ContextBudget("claude-sonnet-4", buffer_tokens=200_000 - available_tokens)
    return InstructionCompressor(budget)


def test_all_parts_fit():
    comp = _make_compressor(100_000)
    result = comp.compress(
        user_instructions="You are helpful.",
        skill_catalog="## Skills\n- greeting",
        history_md="# Thread History\n\n## Run Log\n| # |\n|---|\n| 1 |",
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
    result = comp.compress(
        user_instructions="Critical instruction that must survive.",
        skill_catalog=None,
        history_md="# Thread History\n" + "x" * 5000,
        thread_summary=None,
    )
    assert "Critical instruction that must survive" in result


def test_history_stripped_when_over_budget():
    """history_md should be compressed when budget is insufficient."""
    # Very small budget
    comp = _make_compressor(available_tokens=200)
    big_history = (
        "# Thread History\n\n"
        "## Run Log\n"
        "| # | Time | Task | State | Skills |\n"
        "|---|------|------|-------|--------|\n"
        "| 1 | 2026-04-10 | Task one | success | — |\n"
        "\n"
        "## Recent Runs\n\n"
        "### Run 1 — Task one (success)\n"
        "Skills: — | Steps: 3\n\n"
        "Step 1:\n```python\nprint('hello')\n```\n> hello\n\n"
        "Step 2:\n```python\nprint('world')\n```\n> world\n\n"
    )
    result = comp.compress(
        user_instructions="Be helpful.",
        skill_catalog=None,
        history_md=big_history,
        thread_summary=None,
    )
    # Should contain user_instructions
    assert "Be helpful" in result
    # History should be compressed (at least partially retained or fully discarded)
    assert len(result) < len(big_history) + 50


def test_strip_step_details():
    """_strip_step_details should remove code blocks and quote lines."""
    comp = _make_compressor()
    history = (
        "# Thread History\n\n"
        "## Run Log\n| # |\n|---|\n| 1 |\n\n"
        "## Recent Runs\n\n"
        "### Run 1 — Task (success)\n"
        "Skills: — | Steps: 1\n\n"
        "Step 1:\n```python\ncode_here()\n```\n> output here\n\n"
        "**Output:** final result\n"
    )
    stripped = InstructionCompressor._strip_step_details(history)
    assert "### Run 1" in stripped
    assert "code_here" not in stripped
    assert "**Output:**" in stripped


def test_keep_run_log_only():
    history = (
        "# Thread History\n\n## Run Log\n| # |\n|---|\n\n"
        "## Recent Runs\n\ndetails here"
    )
    result = InstructionCompressor._keep_run_log_only(history)
    assert "Run Log" in result
    assert "Recent Runs" not in result


def test_truncate_run_log():
    lines = ["# Thread History", "", "## Run Log"]
    lines.append("| # | Task |")
    lines.append("|---|------|")
    for i in range(20):
        lines.append(f"| {i+1} | Task {i+1} |")
    history = "\n".join(lines)

    result = InstructionCompressor._truncate_run_log(history, max_rows=5)
    # Should keep only the last 5 data rows
    assert "| 20 |" in result
    assert "| 16 |" in result
    assert "| 15 |" not in result
