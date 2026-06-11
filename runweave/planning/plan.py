from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal

from runweave.planning.errors import UnsupportedPlanVersionError
from runweave.planning.metadata import StepMetadata

StepStatus = Literal["pending", "running", "done", "failed", "blocked"]

_PLAN_VERSION = 1


@dataclass
class PlanStep:
    """One step in a Plan.

    Fields beyond ``id``/``title``/``description``/``metadata`` track execution
    state and are mutated by PlanningRuntime as the step runs.
    """

    id: str
    title: str
    description: str
    metadata: StepMetadata
    depends_on: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    status: StepStatus = "pending"
    selected_model_id: str | None = None
    output: str | None = None
    failure_reason: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


@dataclass
class Plan:
    """A multi-step plan produced by PlannerLLM and executed by PlanningRuntime.

    The plan is the single source of truth for "what's next" — PlanningRuntime
    never asks an LLM to decide the next step. ``next_executable()`` picks
    deterministically.
    """

    version: int
    plan_number: int
    task: str
    created_at: str
    steps: list[PlanStep]

    # -- Factory helpers --------------------------------------------------

    @classmethod
    def new(
        cls, task: str, steps: list[PlanStep], plan_number: int = 0
    ) -> "Plan":
        return cls(
            version=_PLAN_VERSION,
            plan_number=plan_number,
            task=task,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            steps=steps,
        )

    # -- Status mutation --------------------------------------------------

    def _step_by_id(self, step_id: str) -> PlanStep:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(f"unknown step id: {step_id}")

    def mark_running(self, step_id: str) -> None:
        s = self._step_by_id(step_id)
        s.status = "running"
        s.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def mark_done(self, step_id: str, output: str) -> None:
        s = self._step_by_id(step_id)
        s.status = "done"
        s.output = output
        s.completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def mark_failed(self, step_id: str, reason: str) -> None:
        s = self._step_by_id(step_id)
        s.status = "failed"
        s.failure_reason = reason
        s.completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def mark_blocked_downstream(self, failed_step_id: str) -> None:
        """Cascade: any pending step depending on a failed step becomes blocked."""
        failed_or_blocked = {failed_step_id}
        # Walk steps in order; downstream dependencies might chain.
        for step in self.steps:
            if step.status == "pending" and any(
                d in failed_or_blocked for d in step.depends_on
            ):
                step.status = "blocked"
                step.failure_reason = (
                    f"upstream step failed/blocked: {','.join(step.depends_on)}"
                )
                failed_or_blocked.add(step.id)

    # -- Query ------------------------------------------------------------

    def next_executable(self) -> PlanStep | None:
        """Return the next pending step whose deps are all done. None if none.

        Picks in declaration order; deterministic.
        """
        done_ids = {s.id for s in self.steps if s.status == "done"}
        for step in self.steps:
            if step.status != "pending":
                continue
            if all(d in done_ids for d in step.depends_on):
                return step
        return None

    def is_terminal(self) -> bool:
        """All steps in a non-progressable state."""
        return all(s.status in {"done", "failed", "blocked"} for s in self.steps)

    # -- JSON IO ----------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        data = json.loads(text)
        if data.get("version") != _PLAN_VERSION:
            raise UnsupportedPlanVersionError(data.get("version"))
        steps = [
            PlanStep(
                id=s["id"],
                title=s["title"],
                description=s["description"],
                metadata=StepMetadata(**s["metadata"]),
                depends_on=list(s.get("depends_on", [])),
                expected_outputs=list(s.get("expected_outputs", [])),
                status=s.get("status", "pending"),
                selected_model_id=s.get("selected_model_id"),
                output=s.get("output"),
                failure_reason=s.get("failure_reason"),
                started_at=s.get("started_at"),
                completed_at=s.get("completed_at"),
            )
            for s in data["steps"]
        ]
        return cls(
            version=data["version"],
            plan_number=data["plan_number"],
            task=data["task"],
            created_at=data["created_at"],
            steps=steps,
        )
