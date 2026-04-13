from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents import AgentMemory


@dataclass
class StepRecord:
    """Record of a single execution step."""

    step_number: int
    code: str | None = None
    output: str | None = None


@dataclass
class RunRecord:
    """Structured record of a single run."""

    run_number: int
    timestamp: str
    task: str
    state: str
    step_count: int
    skills_used: list[str]
    tools_used: list[str]
    steps: list[StepRecord]
    output: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> RunRecord:
        data = json.loads(text)
        data["steps"] = [StepRecord(**s) for s in data["steps"]]
        data.setdefault("tools_used", [])
        return cls(**data)

    def to_markdown(self) -> str:
        """Generate a human-readable markdown report."""
        lines = [
            f"# Run {self.run_number} — {self.timestamp}",
            "",
            f"**Task:** {self.task}",
            f"**State:** {self.state} | **Steps:** {self.step_count}",
        ]
        if self.skills_used:
            lines.append(f"**Skills:** {', '.join(self.skills_used)}")
        if self.tools_used:
            lines.append(f"**Tools:** {', '.join(self.tools_used)}")
        lines.append("")

        for step in self.steps:
            lines.append(f"## Step {step.step_number}")
            if step.code:
                lines.append("```python")
                lines.append(step.code)
                lines.append("```")
            if step.output:
                lines.append(f"> {step.output}")
            lines.append("")

        lines.append(f"## Output")
        lines.append(str(self.output))
        return "\n".join(lines)


# -- Shared rendering utilities -----------------------------------------------

# Max lines of code per step in rendered history
MAX_CODE_LINES = 10
# Default number of recent runs to show
DEFAULT_RECENT_COUNT = 3


def _escape_cell(text: str) -> str:
    """Escape text for markdown table cells and single-line contexts."""
    return text.replace("|", "\\|").replace("\n", " ")


def render_run_log(records: list[RunRecord]) -> str:
    """Render the Run Log markdown table."""
    lines = [
        "# Thread History",
        "",
        "## Run Log",
        "| # | Time | Task | State | Skills | Tools |",
        "|---|------|------|-------|--------|-------|",
    ]
    for r in records:
        time_short = r.timestamp[:10] if len(r.timestamp) >= 10 else r.timestamp
        skills = ", ".join(r.skills_used) if r.skills_used else "—"
        tools = ", ".join(r.tools_used) if r.tools_used else "—"
        task_short = r.task[:60] + "..." if len(r.task) > 60 else r.task
        lines.append(
            f"| {r.run_number} | {time_short} "
            f"| {_escape_cell(task_short)} "
            f"| {r.state} | {skills} | {tools} |"
        )
    return "\n".join(lines)


def render_recent_runs(
    records: list[RunRecord],
    count: int = DEFAULT_RECENT_COUNT,
    include_steps: bool = True,
) -> str:
    """Render the Recent Runs section.

    When *include_steps* is True, full step details (code + output) are shown.
    When False, only run headers with summary metadata are rendered.
    """
    recent = records[-count:]
    lines: list[str] = ["## Recent Runs", ""]
    for r in reversed(recent):
        skills_str = ", ".join(r.skills_used) if r.skills_used else "—"
        tools_str = ", ".join(r.tools_used) if r.tools_used else "—"
        task_safe = _escape_cell(r.task[:50])
        lines.append(f"### Run {r.run_number} — {task_safe} ({r.state})")
        lines.append(
            f"Skills: {skills_str} | Tools: {tools_str} | Steps: {r.step_count}"
        )

        if include_steps:
            lines.append("")
            for step in r.steps:
                lines.append(f"Step {step.step_number}:")
                if step.code:
                    code_lines = step.code.split("\n")
                    lines.append("```python")
                    lines.extend(code_lines[:MAX_CODE_LINES])
                    if len(code_lines) > MAX_CODE_LINES:
                        lines.append(
                            f"# ... ({len(code_lines) - MAX_CODE_LINES} more lines, "
                            f"use read_run_detail({r.run_number}) for full code)"
                        )
                    lines.append("```")
                if step.output:
                    output_text = step.output[:500]
                    if len(step.output) > 500:
                        output_text += "..."
                    for quote_line in output_text.split("\n"):
                        lines.append(f"> {quote_line}")
                lines.append("")

        output_safe = _escape_cell(str(r.output)[:200])
        lines.append(f"**Output:** {output_safe}")
        lines.append("")
    return "\n".join(lines)


def _count_action_steps(steps: list) -> int:
    """Count only ActionStep instances in a step list."""
    from smolagents.memory import ActionStep as _ActionStep

    return sum(1 for s in steps if isinstance(s, _ActionStep))


def extract_run_record(
    run_number: int,
    task: str,
    memory: AgentMemory,
    state: str,
    skills_used: list[str],
    output: Any,
    tools_used: list[str] | None = None,
) -> RunRecord:
    """Extract a structured RunRecord from agent.memory."""
    from smolagents.memory import ActionStep

    step_records: list[StepRecord] = []
    for step in memory.steps:
        if not isinstance(step, ActionStep):
            continue
        step_records.append(
            StepRecord(
                step_number=step.step_number,
                code=step.code_action,
                output=step.observations,
            )
        )

    return RunRecord(
        run_number=run_number,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        task=task,
        state=state,
        step_count=len(step_records),
        skills_used=skills_used,
        tools_used=tools_used or [],
        steps=step_records,
        output=str(output),
    )
