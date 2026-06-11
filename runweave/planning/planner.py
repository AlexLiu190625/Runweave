from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from smolagents.models import ChatMessage, MessageRole

from runweave.planning.errors import PlannerOutputError
from runweave.planning.metadata import StepMetadata
from runweave.planning.plan import Plan, PlanStep

if TYPE_CHECKING:
    from smolagents.models import Model


@dataclass
class ThreadContext:
    """Context passed to the planner LLM. Distinct from InstructionCompressor
    output — the planner needs different framing than the step CodeAgents.
    """

    key_facts: str | None = None
    summary: str | None = None
    recent_runs_brief: str | None = None
    skill_catalog: str | None = None


_VALID_KINDS = {"read", "edit", "test", "summarize", "research"}
_VALID_DIFFICULTY = {"low", "medium", "high"}
_VALID_STYLES = {"careful", "exploratory", "precise"}


_PLANNER_SYSTEM = (
    "You are a planner that decomposes a user task into a sequence of "
    "executable steps. Each step will be executed by a separate code-writing "
    "agent on a different model chosen by metadata.\n"
    "\n"
    "Return ONLY a JSON object matching this schema:\n"
    "{\n"
    '  "steps": [\n'
    "    {\n"
    '      "id": "s1",\n'
    '      "title": "short human-readable title",\n'
    '      "description": "full task prompt for the step agent — explicit and self-contained",\n'
    '      "metadata": {\n'
    '        "kind": one of read|edit|test|summarize|research,\n'
    '        "needs_long_context": bool,\n'
    '        "needs_structured_output": bool,\n'
    '        "needs_tools": bool,\n'
    '        "difficulty": one of low|medium|high,\n'
    '        "style": one of careful|exploratory|precise\n'
    "      },\n"
    '      "depends_on": ["s0", ...] (other step ids that must complete first),\n'
    '      "expected_outputs": ["filename.py", ...] (workspace files that must exist after the step)\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "\n"
    "Rules:\n"
    "  - Step ids are 's1', 's2', ... in declaration order\n"
    "  - depends_on must reference earlier step ids only\n"
    "  - expected_outputs lists workspace-relative filenames as success criteria\n"
    "  - Do not wrap output in code fences. Output ONLY the JSON."
)


class PlannerLLM:
    """Single-call planner: produces or revises a Plan via one LLM round-trip.

    On parse failure the model is asked once to retry; further failures raise
    PlannerOutputError so the caller can surface a clean error rather than
    proceeding without a plan.
    """

    MAX_PARSE_RETRIES = 1

    def __init__(self, model: "Model") -> None:
        self.model = model

    # -- Public API -------------------------------------------------------

    def plan(self, task: str, ctx: ThreadContext) -> Plan:
        user_prompt = self._compose_plan_prompt(task, ctx)
        steps = self._invoke_with_retry(user_prompt)
        return Plan.new(task=task, steps=steps)

    def replan(
        self,
        previous_plan: Plan,
        failure_summary: str,
        ctx: ThreadContext,
    ) -> Plan:
        user_prompt = self._compose_replan_prompt(previous_plan, failure_summary, ctx)
        steps = self._invoke_with_retry(user_prompt)
        return Plan.new(
            task=previous_plan.task,
            steps=steps,
            plan_number=previous_plan.plan_number,
        )

    # -- Prompt assembly --------------------------------------------------

    def _compose_plan_prompt(self, task: str, ctx: ThreadContext) -> str:
        parts = [f"## User Task\n{task}"]
        if ctx.key_facts:
            parts.append(f"## Key Facts (from prior runs)\n{ctx.key_facts}")
        if ctx.summary:
            parts.append(f"## Thread Summary\n{ctx.summary}")
        if ctx.recent_runs_brief:
            parts.append(f"## Recent Runs\n{ctx.recent_runs_brief}")
        if ctx.skill_catalog:
            parts.append(f"## Available Skills\n{ctx.skill_catalog}")
        parts.append(
            "Decompose this task into steps following the schema in the "
            "system prompt. Output JSON only."
        )
        return "\n\n".join(parts)

    def _compose_replan_prompt(
        self, prev: Plan, failure_summary: str, ctx: ThreadContext
    ) -> str:
        parts = [
            f"## User Task\n{prev.task}",
            f"## Previous Plan (with failures)\n{prev.to_json()}",
            f"## Failure Summary\n{failure_summary}",
        ]
        if ctx.key_facts:
            parts.append(f"## Key Facts\n{ctx.key_facts}")
        parts.append(
            "Produce a revised plan. Keep completed steps in the new plan as "
            "status=done. Adjust remaining steps to recover from the failures. "
            "Output JSON only."
        )
        return "\n\n".join(parts)

    # -- LLM invocation ---------------------------------------------------

    def _invoke_with_retry(self, user_prompt: str) -> list[PlanStep]:
        last_error: PlannerOutputError | None = None
        attempt_prompt = user_prompt
        for attempt in range(1 + self.MAX_PARSE_RETRIES):
            response = self.model(
                messages=[
                    ChatMessage(role=MessageRole.SYSTEM, content=_PLANNER_SYSTEM),
                    ChatMessage(role=MessageRole.USER, content=attempt_prompt),
                ]
            )
            try:
                return self._parse_steps(response.content)
            except PlannerOutputError as e:
                last_error = e
                # On retry, append the parse error to the prompt so the model
                # knows what went wrong.
                attempt_prompt = (
                    user_prompt
                    + f"\n\nYour previous output was rejected: {e.reason}. "
                    "Output ONLY valid JSON matching the schema."
                )
        assert last_error is not None
        raise last_error

    # -- Parsing ----------------------------------------------------------

    def _parse_steps(self, raw: str) -> list[PlanStep]:
        text = _extract_json(raw).strip()
        if not text:
            raise PlannerOutputError("empty output", raw)
        try:
            data: Any = json.loads(text)
        except json.JSONDecodeError as e:
            raise PlannerOutputError(f"invalid JSON: {e}", raw) from e

        if not isinstance(data, dict) or "steps" not in data:
            raise PlannerOutputError("missing 'steps' field", raw)

        raw_steps = data["steps"]
        if not isinstance(raw_steps, list):
            raise PlannerOutputError("'steps' must be a list", raw)

        steps: list[PlanStep] = []
        for i, s in enumerate(raw_steps):
            if not isinstance(s, dict):
                raise PlannerOutputError(f"step #{i} is not an object", raw)
            try:
                meta_dict = s["metadata"]
                kind = meta_dict["kind"]
                difficulty = meta_dict["difficulty"]
                style = meta_dict["style"]
            except (KeyError, TypeError) as e:
                raise PlannerOutputError(f"step #{i} metadata missing field: {e}", raw)

            if kind not in _VALID_KINDS:
                raise PlannerOutputError(
                    f"step #{i} kind={kind!r} not in {sorted(_VALID_KINDS)}", raw
                )
            if difficulty not in _VALID_DIFFICULTY:
                raise PlannerOutputError(
                    f"step #{i} difficulty={difficulty!r} not in {sorted(_VALID_DIFFICULTY)}",
                    raw,
                )
            if style not in _VALID_STYLES:
                raise PlannerOutputError(
                    f"step #{i} style={style!r} not in {sorted(_VALID_STYLES)}", raw
                )

            metadata = StepMetadata(
                kind=kind,
                needs_long_context=bool(meta_dict.get("needs_long_context", False)),
                needs_structured_output=bool(
                    meta_dict.get("needs_structured_output", False)
                ),
                needs_tools=bool(meta_dict.get("needs_tools", False)),
                difficulty=difficulty,
                style=style,
            )
            steps.append(
                PlanStep(
                    id=str(s.get("id") or f"s{i + 1}"),
                    title=str(s.get("title", "")),
                    description=str(s.get("description", "")),
                    metadata=metadata,
                    depends_on=list(s.get("depends_on", [])),
                    expected_outputs=list(s.get("expected_outputs", [])),
                    status=s.get("status", "pending"),
                    output=s.get("output"),
                    failure_reason=s.get("failure_reason"),
                    selected_model_id=s.get("selected_model_id"),
                )
            )
        return steps


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```$", re.DOTALL)


def _extract_json(text: str) -> str:
    """Extract a JSON object from possibly-noisy model output.

    Strategy (first match wins):
      1. Strict markdown code fence ``` or ```json wrapping the JSON
      2. Substring from first '{' to last '}'
      3. Raw stripped text (downstream JSON parser decides if it's valid)
    """
    stripped = text.strip()
    m = _FENCE_RE.match(stripped)
    if m:
        return m.group(1).strip()
    first = stripped.find("{")
    last = stripped.rfind("}")
    if 0 <= first < last:
        return stripped[first:last + 1]
    return stripped
