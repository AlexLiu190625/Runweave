"""Tests for PlannerLLM."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from runweave.planning import PlannerLLM, PlannerOutputError, ThreadContext


class MockModel:
    """Returns scripted responses in FIFO order."""

    model_id = "test-planner"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list] = []

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        content = self.replies.pop(0) if self.replies else ""
        return SimpleNamespace(content=content)


def _valid_plan_json(task_step_count: int = 2) -> str:
    steps = []
    for i in range(task_step_count):
        steps.append({
            "id": f"s{i + 1}",
            "title": f"Step {i + 1}",
            "description": f"Do step {i + 1}",
            "metadata": {
                "kind": "edit",
                "needs_long_context": False,
                "needs_structured_output": False,
                "needs_tools": True,
                "difficulty": "medium",
                "style": "precise",
            },
            "depends_on": [f"s{i}"] if i > 0 else [],
            "expected_outputs": [f"out_{i + 1}.py"],
        })
    return json.dumps({"steps": steps})


def test_plan_parses_valid_json() -> None:
    model = MockModel([_valid_plan_json(2)])
    planner = PlannerLLM(model)
    plan = planner.plan("build CLI", ThreadContext())
    assert plan.task == "build CLI"
    assert len(plan.steps) == 2
    assert plan.steps[0].id == "s1"
    assert plan.steps[1].depends_on == ["s1"]


def test_plan_retries_on_parse_error() -> None:
    model = MockModel([
        "not json at all",
        _valid_plan_json(1),
    ])
    planner = PlannerLLM(model)
    plan = planner.plan("task", ThreadContext())
    assert len(plan.steps) == 1
    assert len(model.calls) == 2  # one failed parse, then retry


def test_plan_raises_after_max_retries() -> None:
    model = MockModel(["garbage", "still garbage"])
    planner = PlannerLLM(model)
    with pytest.raises(PlannerOutputError):
        planner.plan("task", ThreadContext())


def test_plan_rejects_invalid_literal_value() -> None:
    bad = json.dumps({
        "steps": [{
            "id": "s1",
            "title": "x",
            "description": "x",
            "metadata": {
                "kind": "UNKNOWN_KIND",          # <-- invalid
                "needs_long_context": False,
                "needs_structured_output": False,
                "needs_tools": False,
                "difficulty": "medium",
                "style": "precise",
            },
            "depends_on": [],
            "expected_outputs": [],
        }],
    })
    model = MockModel([bad, bad])  # retry also fails
    planner = PlannerLLM(model)
    with pytest.raises(PlannerOutputError):
        planner.plan("task", ThreadContext())


def test_replan_passes_previous_plan_and_failure_to_llm() -> None:
    initial_model = MockModel([_valid_plan_json(2)])
    planner = PlannerLLM(initial_model)
    plan = planner.plan("task", ThreadContext())
    # Mark one as failed so the replan call has something to react to
    plan.mark_failed("s1", "boom")

    replan_model = MockModel([_valid_plan_json(1)])
    planner.model = replan_model
    planner.replan(plan, failure_summary="s1 failed: boom", ctx=ThreadContext())

    # The user-side prompt should contain both the previous plan JSON and
    # the failure summary so the planner LLM has context to recover.
    user_msg = replan_model.calls[0][1].content
    assert "s1 failed: boom" in user_msg
    assert "Previous Plan" in user_msg
