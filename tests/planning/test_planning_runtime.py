"""Integration tests for PlanningRuntime.

PlanningRuntime is tested by monkey-patching ``_execute_step`` so the
deterministic control flow (planner → router → step → outcome check → replan)
can be exercised without spinning up real smolagents.CodeAgents.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runweave.planning import (
    ModelProfile,
    NoCompatibleModelError,
    PlanningRuntime,
    Router,
)
from runweave.planning.runtime import _StepResult, _render_plan_for_injection


class MockModel:
    """LLM mock returning scripted FIFO responses."""

    model_id = "planner-model"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list] = []

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        content = self.replies.pop(0) if self.replies else ""
        return SimpleNamespace(content=content)


def _step_dict(
    step_id: str,
    *,
    kind: str = "edit",
    needs_tools: bool = False,
    needs_structured_output: bool = False,
    needs_long_context: bool = False,
    difficulty: str = "medium",
    style: str = "precise",
    depends_on: list[str] | None = None,
    expected_outputs: list[str] | None = None,
) -> dict:
    return {
        "id": step_id,
        "title": f"Title {step_id}",
        "description": f"Do {step_id}",
        "metadata": {
            "kind": kind,
            "needs_long_context": needs_long_context,
            "needs_structured_output": needs_structured_output,
            "needs_tools": needs_tools,
            "difficulty": difficulty,
            "style": style,
        },
        "depends_on": depends_on or [],
        "expected_outputs": expected_outputs or [],
    }


def _plan_json(*step_kwargs_list) -> str:
    steps = [_step_dict(**kw) for kw in step_kwargs_list]
    return json.dumps({"steps": steps})


def _fake_model(model_id: str):
    return SimpleNamespace(model_id=model_id)


def _profile(model_id: str = "step-model") -> ModelProfile:
    return ModelProfile(
        model=_fake_model(model_id),
        context_window=200_000,
        supports_tools=True,
        supports_structured_output=True,
        coding_score=0.85,
        long_context_score=0.7,
        latency="medium",
        cost_tier="medium",
    )


def _make_runtime(
    tmp_path: Path,
    planner_replies: list[str],
    profiles: list[ModelProfile] | None = None,
    **kwargs,
) -> PlanningRuntime:
    return PlanningRuntime(
        planner_model=MockModel(planner_replies),
        models=profiles or [_profile()],
        base_dir=tmp_path,
        **kwargs,
    )


def _scripted_step_results(*results: _StepResult):
    """Return a stub for _execute_step that hands out scripted results FIFO."""
    iterator = iter(results)
    captured: list[dict] = []

    def stub(self, thread, plan, step, chosen, tool_names):
        captured.append({
            "step_id": step.id,
            "chosen_model_id": chosen.model_id,
            "step_description": step.description,
        })
        try:
            return next(iterator)
        except StopIteration:
            return _StepResult(
                output=f"default-{step.id}",
                state="success",
                tools_used=[],
                skills_used=[],
                token_usage={"input_tokens": 10, "output_tokens": 20},
                elapsed_seconds=0.1,
            )

    return stub, captured


def _ok_step_result(output: str = "ok") -> _StepResult:
    return _StepResult(
        output=output,
        state="success",
        tools_used=[],
        skills_used=[],
        token_usage={"input_tokens": 10, "output_tokens": 20},
        elapsed_seconds=0.1,
    )


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_single_step_plan_succeeds(tmp_path, monkeypatch) -> None:
    rt = _make_runtime(tmp_path, [_plan_json({"step_id": "s1"})])
    stub, captured = _scripted_step_results(_ok_step_result("step1 done"))
    monkeypatch.setattr(PlanningRuntime, "_execute_step", stub)

    result = rt.run("build CLI")

    assert result.state == "success"
    assert "step1 done" in result.output
    assert len(captured) == 1
    assert captured[0]["step_id"] == "s1"


def test_multi_step_with_dependencies(tmp_path, monkeypatch) -> None:
    rt = _make_runtime(
        tmp_path,
        [_plan_json(
            {"step_id": "s1"},
            {"step_id": "s2", "depends_on": ["s1"]},
            {"step_id": "s3", "depends_on": ["s2"]},
        )],
    )
    stub, captured = _scripted_step_results(
        _ok_step_result("a"), _ok_step_result("b"), _ok_step_result("c")
    )
    monkeypatch.setattr(PlanningRuntime, "_execute_step", stub)

    result = rt.run("task")

    assert result.state == "success"
    # Topological order honored
    assert [c["step_id"] for c in captured] == ["s1", "s2", "s3"]


# ---------------------------------------------------------------------------
# Replan
# ---------------------------------------------------------------------------


def test_failed_step_triggers_replan(tmp_path, monkeypatch) -> None:
    """When a step fails, planner.replan is invoked and a recovery step runs."""
    rt = _make_runtime(
        tmp_path,
        [
            _plan_json({"step_id": "s1"}, {"step_id": "s2", "depends_on": ["s1"]}),
            _plan_json({"step_id": "s1b"}),  # recovery plan
        ],
    )
    monkeypatch.setattr(PlanningRuntime, "_safe_summary", lambda *a, **kw: None)
    monkeypatch.setattr(PlanningRuntime, "_safe_key_facts", lambda *a, **kw: None)

    stub, captured = _scripted_step_results(
        _StepResult(
            output=None, state="max_steps_error",
            tools_used=[], skills_used=[], token_usage=None, elapsed_seconds=0.1,
        ),
        _ok_step_result("recovered"),
    )
    monkeypatch.setattr(PlanningRuntime, "_execute_step", stub)

    rt.run("task")

    # The recovery step executed
    assert any(c["step_id"] == "s1b" for c in captured)


def test_max_replans_caps_recovery(tmp_path, monkeypatch) -> None:
    """After max_replans the runtime stops calling planner.replan."""
    plan_replies = [_plan_json({"step_id": "s1"})] * 6
    rt = _make_runtime(tmp_path, plan_replies, max_replans=3)
    monkeypatch.setattr(PlanningRuntime, "_safe_summary", lambda *a, **kw: None)
    monkeypatch.setattr(PlanningRuntime, "_safe_key_facts", lambda *a, **kw: None)
    # Every step result is a failure.
    fail = _StepResult(
        output=None, state="failed",
        tools_used=[], skills_used=[], token_usage=None, elapsed_seconds=0.1,
    )
    stub, captured = _scripted_step_results(fail, fail, fail, fail, fail, fail, fail, fail)
    monkeypatch.setattr(PlanningRuntime, "_execute_step", stub)

    rt.run("task")

    # 1 initial plan + 3 replans = 4 step executions (each iteration consumes
    # one step). Beyond max_replans we stop replanning, so iteration also stops.
    assert len(captured) == 1 + 3


# ---------------------------------------------------------------------------
# Router integration
# ---------------------------------------------------------------------------


def test_router_called_per_step(tmp_path, monkeypatch) -> None:
    """Each step records selected_model_id in the archived plan."""
    cheap = ModelProfile(
        model=_fake_model("cheap-model"),
        context_window=200_000, supports_tools=True, supports_structured_output=True,
        coding_score=0.7, long_context_score=0.65,
        latency="low", cost_tier="low",
    )
    premium = ModelProfile(
        model=_fake_model("premium-model"),
        context_window=200_000, supports_tools=True, supports_structured_output=True,
        coding_score=0.95, long_context_score=0.9,
        latency="high", cost_tier="high",
    )
    rt = _make_runtime(
        tmp_path,
        [_plan_json(
            {"step_id": "s1", "kind": "read", "difficulty": "low", "style": "exploratory"},
            {"step_id": "s2", "kind": "edit", "difficulty": "high", "style": "precise"},
        )],
        profiles=[cheap, premium],
    )
    monkeypatch.setattr(PlanningRuntime, "_safe_summary", lambda *a, **kw: None)
    monkeypatch.setattr(PlanningRuntime, "_safe_key_facts", lambda *a, **kw: None)
    stub, _ = _scripted_step_results(_ok_step_result("a"), _ok_step_result("b"))
    monkeypatch.setattr(PlanningRuntime, "_execute_step", stub)

    rt.run("task")

    archived = list((tmp_path / "threads").rglob("plans/plan-001.json"))
    assert len(archived) == 1
    plan = json.loads(archived[0].read_text())
    selected = {s["id"]: s["selected_model_id"] for s in plan["steps"]}
    # Easy read step → cheap model (no coding boost, low cost wins)
    assert selected["s1"] == "cheap-model"
    # Hard edit step → premium (coding boost + difficulty*tier dominates)
    assert selected["s2"] == "premium-model"


def test_planning_runtime_propagates_router_error(tmp_path, monkeypatch) -> None:
    """If Router cannot find any compatible model, the error surfaces."""
    incompatible = _profile()
    incompatible.supports_tools = False
    rt = _make_runtime(
        tmp_path,
        [_plan_json({"step_id": "s1", "needs_tools": True})],
        profiles=[incompatible],
    )
    monkeypatch.setattr(
        PlanningRuntime, "_execute_step",
        lambda *a, **kw: _ok_step_result(),
    )
    with pytest.raises(NoCompatibleModelError):
        rt.run("task")


# ---------------------------------------------------------------------------
# Outcome checks
# ---------------------------------------------------------------------------


def test_expected_outputs_missing_marks_failed(tmp_path, monkeypatch) -> None:
    """A step declaring expected_outputs that aren't created → marked failed."""
    rt = _make_runtime(
        tmp_path,
        [
            _plan_json({"step_id": "s1", "expected_outputs": ["never_created.py"]}),
            _plan_json({"step_id": "s_recover"}),  # replan
        ],
    )
    stub, _ = _scripted_step_results(_ok_step_result("said ok"), _ok_step_result("done"))
    monkeypatch.setattr(PlanningRuntime, "_execute_step", stub)

    rt.run("task")

    archived = list((tmp_path / "threads").rglob("plans/plan-*.json"))
    # At least the original (failed) plan is archived
    assert len(archived) >= 1


# ---------------------------------------------------------------------------
# Plan archive + run record + key_facts
# ---------------------------------------------------------------------------


def test_plan_archived_to_plans_dir(tmp_path, monkeypatch) -> None:
    rt = _make_runtime(tmp_path, [_plan_json({"step_id": "s1"})])
    monkeypatch.setattr(
        PlanningRuntime, "_execute_step",
        lambda *a, **kw: _ok_step_result(),
    )
    # Stub summary + key_facts to avoid extra LLM
    monkeypatch.setattr(PlanningRuntime, "_safe_summary", lambda *a, **kw: "summary")
    monkeypatch.setattr(PlanningRuntime, "_safe_key_facts", lambda *a, **kw: "- fact")

    result = rt.run("task")

    # plan.json removed (archived)
    threads_dirs = list((tmp_path / "threads").iterdir())
    assert len(threads_dirs) == 1
    td = threads_dirs[0]
    assert not (td / "plan.json").exists()
    assert (td / "plans" / "plan-001.json").is_file()
    assert result.thread_id == td.name


def test_aggregated_run_record_single_entry(tmp_path, monkeypatch) -> None:
    """A PlanningRuntime.run() adds exactly one row to HISTORY/runs."""
    rt = _make_runtime(
        tmp_path,
        [_plan_json({"step_id": "s1"}, {"step_id": "s2", "depends_on": ["s1"]})],
    )
    stub, _ = _scripted_step_results(_ok_step_result(), _ok_step_result())
    monkeypatch.setattr(PlanningRuntime, "_execute_step", stub)
    monkeypatch.setattr(PlanningRuntime, "_safe_summary", lambda *a, **kw: "s")
    monkeypatch.setattr(PlanningRuntime, "_safe_key_facts", lambda *a, **kw: "k")

    rt.run("task")

    threads_dirs = list((tmp_path / "threads").iterdir())
    td = threads_dirs[0]
    run_files = list((td / "runs").glob("run-*.json"))
    assert len(run_files) == 1  # ONE row, not per-step


def test_summary_and_key_facts_generated_after_run(tmp_path, monkeypatch) -> None:
    """PlanningRuntime writes summary.txt and key_facts.md just like Runtime."""
    rt = _make_runtime(tmp_path, [_plan_json({"step_id": "s1"})])
    monkeypatch.setattr(
        PlanningRuntime, "_execute_step",
        lambda *a, **kw: _ok_step_result("step output"),
    )
    monkeypatch.setattr(
        PlanningRuntime, "_safe_summary", lambda *a, **kw: "narrative summary"
    )
    monkeypatch.setattr(
        PlanningRuntime, "_safe_key_facts", lambda *a, **kw: "- [run 1] goal: X"
    )

    rt.run("task")

    td = next((tmp_path / "threads").iterdir())
    assert (td / "summary.txt").read_text() == "narrative summary"
    assert "[run 1] goal: X" in (td / "key_facts.md").read_text()


# ---------------------------------------------------------------------------
# plan-injection text
# ---------------------------------------------------------------------------


def test_plan_text_injected_into_step_agent_system_prompt() -> None:
    """The plan-rendering helper produces a text suitable for injection."""
    from runweave.planning import Plan, PlanStep, StepMetadata

    md = StepMetadata(
        kind="edit", needs_long_context=False, needs_structured_output=False,
        needs_tools=True, difficulty="medium", style="precise",
    )
    plan = Plan.new(
        task="build X",
        steps=[
            PlanStep(id="s1", title="A", description="...", metadata=md),
            PlanStep(id="s2", title="B", description="...", metadata=md,
                     depends_on=["s1"]),
        ],
    )
    plan.mark_done("s1", "ok")
    plan.steps[0].selected_model_id = "haiku"
    plan.steps[1].selected_model_id = "sonnet"

    text = _render_plan_for_injection(plan, current_step_id="s2")
    assert "Task: build X" in text
    assert "[s1] A" in text
    assert "[s2] B" in text
    assert "current step" in text
    assert "haiku" in text
    assert "sonnet" in text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_plan_completes_immediately(tmp_path, monkeypatch) -> None:
    """Planner returns no steps → run() completes without invoking step execution."""
    rt = _make_runtime(tmp_path, [json.dumps({"steps": []})])
    monkeypatch.setattr(PlanningRuntime, "_safe_summary", lambda *a, **kw: None)
    monkeypatch.setattr(PlanningRuntime, "_safe_key_facts", lambda *a, **kw: None)

    called = []
    monkeypatch.setattr(
        PlanningRuntime, "_execute_step",
        lambda *a, **kw: called.append(1) or _ok_step_result(),
    )

    result = rt.run("nothing-to-do")

    assert called == []  # no step executions
    assert result.step_count == 0


def test_orphan_plan_renamed_on_next_run(tmp_path, monkeypatch) -> None:
    """If plan.json exists at run start (prior crash), it gets moved to orphan."""
    # First, fake a thread with a stale plan.json
    threads_dir = tmp_path / "threads" / "thread1"
    threads_dir.mkdir(parents=True)
    (threads_dir / "workspace").mkdir()
    (threads_dir / "runs").mkdir()
    (threads_dir / "meta.json").write_text('{"id": "thread1", "created_at": "2026-04-22T00:00:00+00:00"}')
    (threads_dir / "plan.json").write_text('{"stale": true}')

    rt = _make_runtime(tmp_path, [_plan_json({"step_id": "s1"})])
    monkeypatch.setattr(
        PlanningRuntime, "_execute_step",
        lambda *a, **kw: _ok_step_result(),
    )
    monkeypatch.setattr(PlanningRuntime, "_safe_summary", lambda *a, **kw: None)
    monkeypatch.setattr(PlanningRuntime, "_safe_key_facts", lambda *a, **kw: None)

    rt.run("task", thread_id="thread1")

    orphans = list((threads_dir / "plans").glob("plan-orphan-*.json"))
    assert len(orphans) == 1
    # Content preserved
    assert "stale" in orphans[0].read_text()
