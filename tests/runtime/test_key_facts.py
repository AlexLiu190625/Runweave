"""Unit tests for KeyFactsDistiller."""

from __future__ import annotations

from types import SimpleNamespace

from runweave.runtime.key_facts import KeyFactsDistiller


class MockModel:
    """Minimal stand-in for smolagents.Model.

    Captures the messages passed in and returns a scripted response. Supports
    a single scripted reply or a list consumed FIFO.
    """

    def __init__(self, reply: str | list[str]) -> None:
        self.replies: list[str] = [reply] if isinstance(reply, str) else list(reply)
        self.calls: list[list] = []

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        content = self.replies.pop(0) if self.replies else ""
        return SimpleNamespace(content=content)


# ---------------------------------------------------------------------------
# First-run behavior
# ---------------------------------------------------------------------------


def test_first_run_without_previous():
    model = MockModel("- [run 1] Goal: build calc.py\n- [run 1] Output format: stdin")
    distiller = KeyFactsDistiller(model)

    result = distiller.distill(
        task="Write calc.py that reads stdin",
        output="Created calc.py",
        run_number=1,
        previous_key_facts=None,
    )

    assert "[run 1]" in result
    assert "Goal: build calc.py" in result
    # One LLM call
    assert len(model.calls) == 1
    # User prompt should mention "(empty" for first distillation
    user_msg = model.calls[0][1].content
    assert "empty" in user_msg.lower()


def test_subsequent_run_includes_previous():
    previous = "- [run 1] Goal: build calc.py\n- [run 1] Forbid editing auth/"
    model = MockModel(
        previous + "\n- [run 2] API output format: JSON"
    )
    distiller = KeyFactsDistiller(model)

    distiller.distill(
        task="Add parentheses support",
        output="Updated calc.py",
        run_number=2,
        previous_key_facts=previous,
    )

    user_msg = model.calls[0][1].content
    assert "Forbid editing auth/" in user_msg
    assert "run 2" in user_msg


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_strips_h1_header():
    """Distiller should strip a stray '# Key Facts' header if the model adds one."""
    model = MockModel("# Key Facts\n\n- [run 1] Goal X")
    distiller = KeyFactsDistiller(model)

    result = distiller.distill("t", "o", 1, None)

    assert not result.startswith("# Key Facts")
    assert "[run 1] Goal X" in result


def test_strips_code_fences():
    """Distiller should strip code fences if the model wraps the list."""
    model = MockModel("```markdown\n- [run 1] X\n- [run 1] Y\n```")
    distiller = KeyFactsDistiller(model)

    result = distiller.distill("t", "o", 1, None)

    assert "```" not in result
    assert "[run 1] X" in result
    assert "[run 1] Y" in result


def test_leaves_plain_bullets_unchanged():
    model = MockModel("- [run 1] X\n- [run 2] Y")
    distiller = KeyFactsDistiller(model)

    result = distiller.distill("t", "o", 2, "- [run 1] X")

    assert result == "- [run 1] X\n- [run 2] Y"


# ---------------------------------------------------------------------------
# Input shaping
# ---------------------------------------------------------------------------


def test_truncates_large_output():
    """The distiller should not feed unbounded output text into the prompt."""
    long_output = "x" * 20_000
    model = MockModel("- [run 1] ok")
    distiller = KeyFactsDistiller(model)

    distiller.distill("task", long_output, 1, None)

    user_msg = model.calls[0][1].content
    # Prompt must not contain the full 20k x's
    assert user_msg.count("x") < 20_000
    assert "[truncated]" in user_msg


def test_handles_none_output():
    model = MockModel("- [run 1] ok")
    distiller = KeyFactsDistiller(model)
    result = distiller.distill("task", None, 1, None)
    assert "[run 1] ok" in result


def test_handles_empty_previous_key_facts():
    """Empty-string previous_key_facts should behave like None (first run)."""
    model = MockModel("- [run 1] X")
    distiller = KeyFactsDistiller(model)

    distiller.distill("task", "output", 1, previous_key_facts="   ")

    user_msg = model.calls[0][1].content
    assert "empty" in user_msg.lower()
