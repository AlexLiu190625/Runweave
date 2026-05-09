"""Integration tests covering the Runtime key_facts flow end-to-end."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from smolagents import AgentMemory

from runweave.runtime.runtime import Runtime


class ScriptedModel:
    """Mock model that returns a different response per keyword in the user prompt.

    The first matching keyword wins, so callers can route summary-bound calls
    vs. key_facts-bound calls to distinct replies.
    """

    model_id = "test-model"

    def __init__(self, rules: dict[str, str], default: str = "") -> None:
        self.rules = rules
        self.default = default
        self.calls: list[list] = []

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        user_content = next(
            (m.content for m in messages if getattr(m.role, "value", m.role) == "user"),
            "",
        )
        for keyword, reply in self.rules.items():
            if keyword in user_content:
                return SimpleNamespace(content=reply)
        return SimpleNamespace(content=self.default)


def _make_runtime(tmp_path: Path, model) -> Runtime:
    return Runtime(model=model, base_dir=tmp_path)


def _make_smolagents_result(output: str = "done"):
    """Build a minimal smolagents-compatible run result."""
    return SimpleNamespace(
        output=output,
        state="success",
        steps=[],
        token_usage=None,
        timing=None,
    )


def _make_agent():
    """Build an object with the minimal surface _finalize_run touches on agent."""
    return SimpleNamespace(memory=AgentMemory("system prompt"))


# ---------------------------------------------------------------------------
# First run: key_facts.md is created on disk
# ---------------------------------------------------------------------------


def test_first_run_creates_key_facts_file(tmp_path: Path) -> None:
    model = ScriptedModel(
        rules={
            "Key Facts": "- [run 1] Goal: build X",
            "summary": "run 1 summary",
        },
        default="fallback",
    )
    rt = _make_runtime(tmp_path, model)
    thread = rt.store.create()
    agent = _make_agent()

    rt._finalize_run(
        agent=agent,
        thread=thread,
        task="Build X",
        smolagents_result=_make_smolagents_result("done"),
        tools_used=[],
    )

    assert thread.key_facts_path.is_file()
    content = thread.key_facts_path.read_text()
    assert "[run 1] Goal: build X" in content


# ---------------------------------------------------------------------------
# Second run: distiller sees previous key_facts in its user prompt
# ---------------------------------------------------------------------------


def test_second_run_distiller_receives_previous(tmp_path: Path) -> None:
    model = ScriptedModel(
        rules={
            "Key Facts": "- [run 1] X\n- [run 2] Y",
            "summary": "narrative",
        },
    )
    rt = _make_runtime(tmp_path, model)
    thread = rt.store.create()

    # Seed a previous key_facts file
    thread.key_facts_path.write_text("- [run 1] X")
    thread.summary_path.write_text("prev summary")

    rt._finalize_run(
        agent=_make_agent(),
        thread=thread,
        task="Do more",
        smolagents_result=_make_smolagents_result("output 2"),
        tools_used=[],
    )

    # The distiller must have seen "- [run 1] X" in its prompt
    distiller_call = next(
        call for call in model.calls
        if any("Key Facts" in m.content for m in call)
    )
    user_msg = distiller_call[1].content
    assert "[run 1] X" in user_msg


# ---------------------------------------------------------------------------
# Fault isolation: one side fails, other side succeeds
# ---------------------------------------------------------------------------


def test_distiller_failure_does_not_break_summary(tmp_path: Path) -> None:
    class FailingOnKeyFacts(ScriptedModel):
        def __call__(self, messages, **kwargs):
            user_content = next(
                (m.content for m in messages
                 if getattr(m.role, "value", m.role) == "user"),
                "",
            )
            if "Key Facts" in user_content:
                raise RuntimeError("distiller boom")
            self.calls.append(messages)
            return SimpleNamespace(content="summary ok")

    model = FailingOnKeyFacts(rules={})
    rt = _make_runtime(tmp_path, model)
    thread = rt.store.create()

    rt._finalize_run(
        agent=_make_agent(),
        thread=thread,
        task="task",
        smolagents_result=_make_smolagents_result("out"),
        tools_used=[],
    )

    # Summary persisted; key_facts absent (no previous either)
    assert thread.summary_path.is_file()
    assert thread.summary_path.read_text() == "summary ok"
    assert not thread.key_facts_path.is_file()


def test_summary_failure_does_not_break_distiller(tmp_path: Path) -> None:
    class FailingOnSummary(ScriptedModel):
        def __call__(self, messages, **kwargs):
            user_content = next(
                (m.content for m in messages
                 if getattr(m.role, "value", m.role) == "user"),
                "",
            )
            if "Key Facts" in user_content:
                self.calls.append(messages)
                return SimpleNamespace(content="- [run 1] ok")
            raise RuntimeError("summary boom")

    model = FailingOnSummary(rules={})
    rt = _make_runtime(tmp_path, model)
    thread = rt.store.create()

    rt._finalize_run(
        agent=_make_agent(),
        thread=thread,
        task="task",
        smolagents_result=_make_smolagents_result("out"),
        tools_used=[],
    )

    assert thread.key_facts_path.is_file()
    assert "[run 1] ok" in thread.key_facts_path.read_text()
    assert not thread.summary_path.is_file()


# ---------------------------------------------------------------------------
# Instruction assembly: key_facts feeds into InstructionCompressor
# ---------------------------------------------------------------------------


def test_collect_instruction_parts_reads_key_facts(tmp_path: Path) -> None:
    model = ScriptedModel(rules={})
    rt = _make_runtime(tmp_path, model)
    thread = rt.store.create()
    thread.key_facts_path.write_text("- [run 1] foo")

    parts = rt._collect_instruction_parts(thread)

    assert parts["key_facts"] == "- [run 1] foo"


def test_collect_instruction_parts_missing_key_facts_returns_none(tmp_path: Path) -> None:
    model = ScriptedModel(rules={})
    rt = _make_runtime(tmp_path, model)
    thread = rt.store.create()

    parts = rt._collect_instruction_parts(thread)

    assert parts["key_facts"] is None
