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
    """单步执行记录。"""

    step_number: int
    code: str | None = None
    output: str | None = None


@dataclass
class RunRecord:
    """一次 run 的结构化记录。"""

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
        """生成人可读的 markdown 详细记录。"""
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


def extract_run_record(
    run_number: int,
    task: str,
    memory: AgentMemory,
    state: str,
    skills_used: list[str],
    output: Any,
    tools_used: list[str] | None = None,
) -> RunRecord:
    """从 agent.memory 中提取结构化的 RunRecord。"""
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
