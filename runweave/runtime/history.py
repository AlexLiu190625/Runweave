from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from smolagents import Tool

from runweave.runtime.run_record import RunRecord

if TYPE_CHECKING:
    pass

# 每步 code 在 HISTORY.md 中的最大行数
_MAX_CODE_LINES = 10
# Recent Runs 中保留的最近 run 数量
_DEFAULT_RECENT_COUNT = 3


class HistoryWriter:
    """管理 per-run 记录文件和 HISTORY.md 索引。"""

    def __init__(
        self,
        runs_dir: Path,
        history_path: Path,
        recent_count: int = _DEFAULT_RECENT_COUNT,
    ) -> None:
        self.runs_dir = runs_dir
        self.history_path = history_path
        self.recent_count = recent_count

    def next_run_number(self) -> int:
        """根据 runs/ 下已有文件计算下一个 run 编号。"""
        existing = list(self.runs_dir.glob("run-*.json"))
        if not existing:
            return 1
        numbers = []
        for p in existing:
            # run-001.json -> 1
            stem = p.stem  # "run-001"
            try:
                numbers.append(int(stem.split("-")[1]))
            except (IndexError, ValueError):
                continue
        return max(numbers) + 1 if numbers else 1

    def save_run(self, record: RunRecord) -> None:
        """写入 runs/run-NNN.json 和 runs/run-NNN.md。"""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"run-{record.run_number:03d}"
        (self.runs_dir / f"{prefix}.json").write_text(
            record.to_json(), encoding="utf-8"
        )
        (self.runs_dir / f"{prefix}.md").write_text(
            record.to_markdown(), encoding="utf-8"
        )

    def generate_history(self) -> None:
        """从 runs/ 目录重新生成 HISTORY.md。"""
        records = self._load_all_records()
        if not records:
            return

        lines: list[str] = []
        lines.append("# Thread History")
        lines.append("")

        # Run Log 表
        lines.append("## Run Log")
        lines.append("| # | Time | Task | State | Skills | Tools |")
        lines.append("|---|------|------|-------|--------|-------|")
        for r in records:
            time_short = r.timestamp[:10] if len(r.timestamp) >= 10 else r.timestamp
            skills = ", ".join(r.skills_used) if r.skills_used else "—"
            tools = ", ".join(r.tools_used) if r.tools_used else "—"
            task_short = r.task[:60] + "..." if len(r.task) > 60 else r.task
            lines.append(
                f"| {r.run_number} | {time_short} | {task_short} | {r.state} | {skills} | {tools} |"
            )
        lines.append("")

        # Recent Runs 详情
        recent = records[-self.recent_count :]
        lines.append("## Recent Runs")
        lines.append("")
        for r in reversed(recent):
            skills_str = ", ".join(r.skills_used) if r.skills_used else "—"
            lines.append(
                f"### Run {r.run_number} — {r.task[:50]} ({r.state})"
            )
            tools_str = ", ".join(r.tools_used) if r.tools_used else "—"
            lines.append(f"Skills: {skills_str} | Tools: {tools_str} | Steps: {r.step_count}")
            lines.append("")
            for step in r.steps:
                lines.append(f"Step {step.step_number}:")
                if step.code:
                    code_lines = step.code.split("\n")
                    lines.append("```python")
                    lines.extend(code_lines[:_MAX_CODE_LINES])
                    if len(code_lines) > _MAX_CODE_LINES:
                        lines.append(
                            f"# ... ({len(code_lines) - _MAX_CODE_LINES} more lines, "
                            f"use read_run_detail({r.run_number}) for full code)"
                        )
                    lines.append("```")
                if step.output:
                    # 截断过长的 output
                    output_text = step.output[:500]
                    if len(step.output) > 500:
                        output_text += "..."
                    lines.append(f"> {output_text}")
                lines.append("")
            lines.append(f"**Output:** {str(r.output)[:200]}")
            lines.append("")

        self.history_path.write_text("\n".join(lines), encoding="utf-8")

    def _load_all_records(self) -> list[RunRecord]:
        """按 run_number 排序加载所有 RunRecord。"""
        records: list[RunRecord] = []
        for json_path in sorted(self.runs_dir.glob("run-*.json")):
            try:
                records.append(
                    RunRecord.from_json(json_path.read_text(encoding="utf-8"))
                )
            except (ValueError, KeyError):
                continue
        records.sort(key=lambda r: r.run_number)
        return records


class ReadRunDetailTool(Tool):
    """按需读取指定 run 编号的详细执行记录。"""

    name = "read_run_detail"
    description = (
        "读取指定 run 编号的详细执行记录，包含完整的代码和输出。"
        "当你需要回顾较早 run 的详细信息时使用。"
    )
    inputs = {
        "run_number": {
            "type": "integer",
            "description": "run 编号（来自 Thread History 的 Run Log）",
        },
    }
    output_type = "string"

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        super().__init__()

    def forward(self, run_number: int) -> str:
        md_path = self.runs_dir / f"run-{run_number:03d}.md"
        if not md_path.is_file():
            return f"错误：未找到 Run {run_number} 的记录。"
        return md_path.read_text(encoding="utf-8")
