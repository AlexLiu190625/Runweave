from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from smolagents import CodeAgent
from smolagents.memory import ActionStep

from runweave.context import ContextBudget, InstructionCompressor, make_context_callback
from runweave.executor.workspace_executor import WorkspaceExecutor
from runweave.planning.errors import PlannerOutputError
from runweave.planning.plan import Plan, PlanStep
from runweave.planning.planner import PlannerLLM, ThreadContext
from runweave.planning.profile import ModelProfile
from runweave.planning.router import Router
from runweave.planning.tracking import TokenUsageTracker, TrackedModel
from runweave.runtime.history import HistoryWriter, ReadRunDetailTool
from runweave.runtime.key_facts import KeyFactsDistiller
from runweave.runtime.memory_io import save_memory
from runweave.runtime.result import RunResult
from runweave.runtime.run_record import RunRecord, StepRecord, _escape_cell
from runweave.runtime.summary import SummaryGenerator
from runweave.runtime.thread import Thread
from runweave.runtime.thread_store import ThreadStore
from runweave.skill.loader import SkillLoader
from runweave.tool.loader import ToolLoader

if TYPE_CHECKING:
    from smolagents.models import Model

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path.home() / ".runweave"

# Cleanup constants (PR 4).
_PLANNER_RECENT_RUN_COUNT = 3
_PLANNER_TASK_PREVIEW_CHARS = 80


@dataclass
class _StepResult:
    output: str | None
    state: str
    tools_used: list[str]
    skills_used: list[str]
    token_usage: dict[str, int] | None
    elapsed_seconds: float


class PlanningRuntime:
    """Orchestrator: planner LLM produces a Plan, Router selects a model per step,
    smolagents.CodeAgent executes each step.

    PlanningRuntime is NOT an agent — it does not call an LLM to decide which
    step is next. The next step is taken deterministically from the plan via
    ``plan.next_executable()``. The only LLM calls are: 1 planner call, 1 call
    per executed step, and at most ``max_replans`` replan calls.

    Known limitations (v0.3):
      - ``expected_outputs`` check verifies file existence only, not
        modification. If an upstream step already created the file and the
        current step declared it as ``expected_output`` but did nothing, the
        check still passes. Workaround: planner should avoid duplicating
        ``expected_outputs`` across steps. v0.4 will add mtime tracking.
    """

    def __init__(
        self,
        planner_model: "Model",
        models: list[ModelProfile],
        *,
        summary_model: "Model | None" = None,
        router: Router | None = None,
        instructions: str | None = None,
        base_dir: Path | None = None,
        additional_authorized_imports: list[str] | None = None,
        skills_dir: Path | None = None,
        tools_dir: Path | None = None,
        context_budget: ContextBudget | None = None,
        step_timeout_seconds: int = 600,
        max_step_iterations: int = 30,
        max_replans: int = 3,
    ) -> None:
        if not models:
            raise ValueError("PlanningRuntime requires at least one ModelProfile")
        self.planner_model = planner_model
        # Cost optimization: a cheaper model can run summary/key_facts while
        # the planner stays strong. Defaults to planner_model (back-compat).
        self.summary_model = summary_model if summary_model is not None else planner_model
        self.models = models
        self.router = router or Router()
        self.instructions = instructions
        self.additional_authorized_imports = additional_authorized_imports or []
        self.store = ThreadStore(base_dir or DEFAULT_BASE_DIR)
        self.skill_loader = SkillLoader(skills_dir) if skills_dir else None
        self.tool_loader = ToolLoader(tools_dir) if tools_dir else None
        # Use the planner model's id as a proxy for the context budget when
        # not explicitly given — the planner is typically the strongest model
        # and represents an upper bound on what the prompt should target.
        self.context_budget = context_budget or ContextBudget(planner_model.model_id)
        self.step_timeout_seconds = step_timeout_seconds
        self.max_step_iterations = max_step_iterations
        self.max_replans = max_replans

    # -- Public API -------------------------------------------------------

    def run(
        self,
        task: str,
        thread_id: str | None = None,
        tool_names: list[str] | None = None,
    ) -> RunResult:
        """Execute a task under planner-driven orchestration."""
        thread = self._load_or_create_thread(thread_id)
        self._handle_orphan_plan(thread)

        # Shared tracker accumulates token usage from planner + summary +
        # key_facts calls. Step CodeAgents are NOT wrapped — their usage
        # comes through _StepResult.token_usage (wrapping would double-count).
        tracker = TokenUsageTracker()
        tracked_planner_model = TrackedModel(self.planner_model, tracker)
        tracked_summary_model = TrackedModel(self.summary_model, tracker)
        self._tracked_summary_model = tracked_summary_model  # used by _safe_*
        planner = PlannerLLM(tracked_planner_model)

        ctx = self._collect_thread_context(thread)

        # 1. Plan (1 LLM call)
        plan = planner.plan(task, ctx)
        plan.plan_number = self._next_plan_number(thread)
        self._write_plan(thread, plan)

        step_outputs: list[str] = []
        aggregated_tools: list[str] = []
        aggregated_skills: list[str] = []
        token_in, token_out = 0, 0
        replan_count = 0

        for _iteration in range(self.max_step_iterations):
            if plan.is_terminal():
                break
            step = plan.next_executable()
            if step is None:
                break

            # 2. Route (0 LLM calls)
            chosen = self.router.select(step.metadata, self.models)
            step.selected_model_id = chosen.model_id
            plan.mark_running(step.id)
            self._write_plan(thread, plan)

            # 3. Execute (1 LLM call via smolagents.CodeAgent)
            step_result = self._execute_step(thread, plan, step, chosen, tool_names)
            step_outputs.append(step_result.output or "")
            aggregated_tools.extend(step_result.tools_used)
            aggregated_skills.extend(step_result.skills_used)
            if step_result.token_usage:
                token_in += step_result.token_usage["input_tokens"]
                token_out += step_result.token_usage["output_tokens"]

            # 4. Deterministic outcome check (no LLM)
            failure = self._check_step_outcome(step, step_result, thread)
            if failure is None:
                plan.mark_done(step.id, step_result.output or "")
            else:
                plan.mark_failed(step.id, failure)
                plan.mark_blocked_downstream(step.id)
            self._write_plan(thread, plan)

            # 5. Replan if any failure AND budget remains.
            # Don't gate on is_terminal — recovery is exactly when terminal
            # state needs unblocking via a fresh plan.
            if (
                any(s.status == "failed" for s in plan.steps)
                and replan_count < self.max_replans
            ):
                failure_summary = self._summarize_failures(plan)
                try:
                    plan = planner.replan(plan, failure_summary, ctx)
                except PlannerOutputError:
                    logger.warning("Replan failed; keeping current plan as-is")
                    break
                plan.plan_number += 1
                replan_count += 1
                self._write_plan(thread, plan)

        # 6. Archive plan
        self._archive_plan(thread, plan)

        # 7. Aggregate single RunRecord and finalize.
        # token_usage is computed INSIDE _finalize_planning_run after summary
        # and key_facts have run, so the tracker has captured all calls.
        return self._finalize_planning_run(
            thread=thread,
            task=task,
            plan=plan,
            step_outputs=step_outputs,
            aggregated_tools=aggregated_tools,
            aggregated_skills=aggregated_skills,
            step_input_tokens=token_in,
            step_output_tokens=token_out,
            tracker=tracker,
        )

    # -- Internal: thread setup ------------------------------------------

    def _load_or_create_thread(self, thread_id: str | None) -> Thread:
        if thread_id and self.store.exists(thread_id):
            return self.store.load(thread_id)
        return self.store.create(thread_id)

    def _handle_orphan_plan(self, thread: Thread) -> None:
        """Rename a stale plan.json (from a prior crashed run) so the new run starts clean."""
        if not thread.plan_path.is_file():
            return
        thread.plans_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        orphan_path = thread.plans_dir / f"plan-orphan-{ts}.json"
        thread.plan_path.rename(orphan_path)
        logger.warning(
            "Orphan plan.json from a prior run moved to %s", orphan_path
        )

    # -- Internal: context for planner -----------------------------------

    def _collect_thread_context(self, thread: Thread) -> ThreadContext:
        key_facts = (
            thread.key_facts_path.read_text().strip()
            if thread.key_facts_path.is_file()
            else None
        )
        summary = (
            thread.summary_path.read_text().strip()
            if thread.summary_path.is_file()
            else None
        )
        recent = self._render_recent_runs_brief(thread)
        skill_catalog = None
        if self.skill_loader:
            cat = self.skill_loader.get_catalog()
            if cat:
                skill_catalog = cat
        return ThreadContext(
            key_facts=key_facts or None,
            summary=summary or None,
            recent_runs_brief=recent,
            skill_catalog=skill_catalog,
        )

    def _render_recent_runs_brief(self, thread: Thread) -> str | None:
        """One-line-per-run brief of the last few runs, or None if no runs exist yet."""
        writer = HistoryWriter(thread.runs_dir, thread.history_path)
        records = writer.load_records()
        if not records:
            return None
        recent = records[-_PLANNER_RECENT_RUN_COUNT:]
        return "\n".join(
            f"- Run {r.run_number}: "
            f"{_escape_cell(r.task[:_PLANNER_TASK_PREVIEW_CHARS])} ({r.state})"
            for r in recent
        )

    # -- Internal: plan I/O ----------------------------------------------

    def _next_plan_number(self, thread: Thread) -> int:
        if not thread.plans_dir.is_dir():
            return 1
        existing = list(thread.plans_dir.glob("plan-*.json"))
        nums: list[int] = []
        for p in existing:
            stem = p.stem
            if not stem.startswith("plan-") or "orphan" in stem:
                continue
            try:
                nums.append(int(stem.split("-")[1]))
            except (IndexError, ValueError):
                continue
        return max(nums) + 1 if nums else 1

    def _write_plan(self, thread: Thread, plan: Plan) -> None:
        thread.plan_path.write_text(plan.to_json(), encoding="utf-8")

    def _archive_plan(self, thread: Thread, plan: Plan) -> None:
        if not thread.plan_path.is_file():
            return
        thread.plans_dir.mkdir(parents=True, exist_ok=True)
        archive_path = thread.plans_dir / f"plan-{plan.plan_number:03d}.json"
        thread.plan_path.rename(archive_path)

    # -- Internal: step execution ----------------------------------------

    def _execute_step(
        self,
        thread: Thread,
        plan: Plan,
        step: PlanStep,
        chosen: ModelProfile,
        tool_names: list[str] | None,
    ) -> _StepResult:
        """Build a CodeAgent for one step and run it.

        Mirrors the assembly logic in Runtime._prepare_run() but is intentionally
        kept separate to avoid coupling PlanningRuntime to Runtime internals.
        """
        executor = WorkspaceExecutor(
            workspace_dir=thread.workspace_dir,
            additional_authorized_imports=self.additional_authorized_imports,
        )

        # Tools: custom + skill + built-in (matches Runtime._prepare_run)
        tools: list = []
        tools_used: list[str] = []
        if self.tool_loader:
            custom = self.tool_loader.get_tools(tool_names)
            tools.extend(custom)
            tools_used = [t.name for t in custom]
        if self.skill_loader:
            tools.extend(self.skill_loader.get_tools())

        tools.append(ReadRunDetailTool(thread.runs_dir))
        from runweave.runtime.workspace_tools import (
            ListFilesTool,
            ReadFileTool,
            WriteFileTool,
        )
        tools.append(WriteFileTool(thread.workspace_dir))
        tools.append(ReadFileTool(thread.workspace_dir))
        tools.append(ListFilesTool(thread.workspace_dir))

        # Instructions: pass plan text into compressor as a new fixed segment
        plan_text = _render_plan_for_injection(plan, current_step_id=step.id)
        instruction_parts = self._collect_instruction_parts(thread)
        instruction_parts["plan"] = plan_text
        compressor = InstructionCompressor(self.context_budget)
        instructions = compressor.compress(**instruction_parts)

        callback = make_context_callback(self.context_budget)
        agent = CodeAgent(
            model=chosen.model,
            tools=tools,
            executor=executor,
            instructions=instructions,
            step_callbacks={ActionStep: callback},
        )

        started = time.monotonic()
        smol_result = agent.run(step.description, return_full_result=True)
        elapsed = time.monotonic() - started

        skills_used: list[str] = []
        if self.skill_loader:
            skills_used = self.skill_loader.load_skill_tool.get_loaded_and_reset()

        return _StepResult(
            output=str(smol_result.output) if smol_result.output is not None else None,
            state=smol_result.state,
            tools_used=tools_used,
            skills_used=skills_used,
            token_usage=(
                smol_result.token_usage.dict() if smol_result.token_usage else None
            ),
            elapsed_seconds=elapsed,
        )

    def _collect_instruction_parts(self, thread: Thread) -> dict[str, Any]:
        """Same shape as Runtime._collect_instruction_parts (sans plan)."""
        skill_catalog = None
        if self.skill_loader:
            cat = self.skill_loader.get_catalog()
            if cat:
                skill_catalog = cat

        history_writer = HistoryWriter(thread.runs_dir, thread.history_path)
        history_records = history_writer.load_records() or None

        thread_summary = None
        if thread.summary_path.is_file():
            s = thread.summary_path.read_text().strip()
            if s:
                thread_summary = s

        key_facts = None
        if thread.key_facts_path.is_file():
            kf = thread.key_facts_path.read_text().strip()
            if kf:
                key_facts = kf

        return {
            "user_instructions": self.instructions,
            "skill_catalog": skill_catalog,
            "history_records": history_records,
            "thread_summary": thread_summary,
            "key_facts": key_facts,
        }

    # -- Internal: deterministic outcome ---------------------------------

    def _check_step_outcome(
        self, step: PlanStep, step_result: _StepResult, thread: Thread
    ) -> str | None:
        """Return failure reason string, or None on success. No LLM calls."""
        if step_result.state != "success":
            return f"agent state: {step_result.state}"
        if step_result.elapsed_seconds > self.step_timeout_seconds:
            return (
                f"timeout: {step_result.elapsed_seconds:.0f}s > "
                f"{self.step_timeout_seconds}s"
            )
        for output_path in step.expected_outputs:
            if not (thread.workspace_dir / output_path).is_file():
                return f"expected output missing: {output_path}"
        return None

    def _summarize_failures(self, plan: Plan) -> str:
        failed = [s for s in plan.steps if s.status == "failed"]
        if not failed:
            return "(no failures)"
        lines = []
        for s in failed:
            lines.append(f"- {s.id} ({s.title}): {s.failure_reason}")
        return "\n".join(lines)

    # -- Internal: finalize ----------------------------------------------

    def _finalize_planning_run(
        self,
        thread: Thread,
        task: str,
        plan: Plan,
        step_outputs: list[str],
        aggregated_tools: list[str],
        aggregated_skills: list[str],
        step_input_tokens: int,
        step_output_tokens: int,
        tracker: TokenUsageTracker,
    ) -> RunResult:
        """Write a single aggregated RunRecord and update summary/key_facts."""
        # Derive top-level state from plan terminal status
        any_failed = any(s.status in {"failed", "blocked"} for s in plan.steps)
        all_done = all(s.status == "done" for s in plan.steps)
        state = "success" if all_done else ("partial" if any_failed else "max_steps_error")

        aggregated_output = "\n\n---\n\n".join(o for o in step_outputs if o)

        # Build a single RunRecord (no per-smolagents-step expansion; the
        # planning steps are summarized differently).
        history_writer = HistoryWriter(thread.runs_dir, thread.history_path)
        run_number = history_writer.next_run_number()

        # Synthesize StepRecords from PlanStep summaries so HISTORY shows
        # what got done at the planning level.
        step_records = [
            StepRecord(
                step_number=i + 1,
                code=f"# planning step {ps.id}: {ps.title} (model: {ps.selected_model_id})",
                output=(ps.output or ps.failure_reason or "")[:500],
            )
            for i, ps in enumerate(plan.steps)
        ]

        run_record = RunRecord(
            run_number=run_number,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            task=task,
            state=state,
            step_count=len(plan.steps),
            skills_used=sorted(set(aggregated_skills)),
            tools_used=sorted(set(aggregated_tools)),
            steps=step_records,
            output=aggregated_output,
        )
        history_writer.save_run(run_record)
        history_writer.generate_history()

        # Generate / update summary + key_facts (using planner model — it's
        # already the strongest model the user provided).
        previous_summary = (
            thread.summary_path.read_text().strip()
            if thread.summary_path.is_file()
            else None
        )
        previous_key_facts = (
            thread.key_facts_path.read_text().strip()
            if thread.key_facts_path.is_file()
            else None
        )
        summary = self._safe_summary(
            task, aggregated_output, previous_summary
        )
        key_facts = self._safe_key_facts(
            task, aggregated_output, previous_key_facts, run_record.run_number
        )
        if summary:
            thread.summary_path.write_text(summary)
        if key_facts:
            thread.key_facts_path.write_text(key_facts)

        # No agent memory to persist for the planning-level run; per-step
        # CodeAgents already managed their own memory in-flight. We do NOT
        # write a unified memory.json (PlanningRuntime's memory model is the
        # plan.json archive itself).

        # Now that summary + key_facts have run, the tracker holds the
        # complete planner + replan + summary + key_facts token usage.
        final_token_usage = {
            "input_tokens": step_input_tokens + tracker.input_tokens,
            "output_tokens": step_output_tokens + tracker.output_tokens,
        }

        return RunResult(
            output=aggregated_output,
            thread_id=thread.id,
            state=state,
            step_count=len(plan.steps),
            token_usage=final_token_usage,
            timing=None,
            summary=summary,
            skills_used=sorted(set(aggregated_skills)),
        )

    def _safe_summary(
        self, task: str, output: str, previous: str | None
    ) -> str | None:
        # Use the tracked summary model so token usage is aggregated. If run()
        # wasn't called (e.g., subclass calls _safe_summary directly), fall
        # back to the unwrapped summary_model.
        model = getattr(self, "_tracked_summary_model", None) or self.summary_model
        try:
            return SummaryGenerator(model).generate(
                task=task, output=output, previous_summary=previous
            )
        except Exception:
            logger.warning("Summary generation failed in planning run", exc_info=True)
            return previous

    def _safe_key_facts(
        self,
        task: str,
        output: str,
        previous: str | None,
        run_number: int,
    ) -> str | None:
        model = getattr(self, "_tracked_summary_model", None) or self.summary_model
        try:
            return KeyFactsDistiller(model).distill(
                task=task,
                output=output,
                run_number=run_number,
                previous_key_facts=previous,
            )
        except Exception:
            logger.warning("Key facts distillation failed in planning run", exc_info=True)
            return previous


def _render_plan_for_injection(plan: Plan, current_step_id: str) -> str:
    """Compact one-line-per-step view of the plan for the step agent's prompt."""
    lines = [f"Task: {plan.task}", "", "Steps:"]
    for s in plan.steps:
        if s.status == "done":
            icon = "✓"
        elif s.id == current_step_id:
            icon = "▶"
        elif s.status == "failed":
            icon = "✗"
        elif s.status == "blocked":
            icon = "⊘"
        else:
            icon = " "
        model_tag = f" ({s.selected_model_id})" if s.selected_model_id else ""
        marker = " <-- current step" if s.id == current_step_id else ""
        lines.append(f"  {icon} [{s.id}] {s.title}{model_tag}{marker}")
    return "\n".join(lines)
