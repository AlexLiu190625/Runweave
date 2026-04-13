from __future__ import annotations

from pathlib import Path

from smolagents import Tool

from runweave.runtime.run_record import (
    DEFAULT_RECENT_COUNT,
    RunRecord,
    render_recent_runs,
    render_run_log,
)


class HistoryWriter:
    """Manage per-run record files and the HISTORY.md index."""

    def __init__(
        self,
        runs_dir: Path,
        history_path: Path,
        recent_count: int = DEFAULT_RECENT_COUNT,
    ) -> None:
        self.runs_dir = runs_dir
        self.history_path = history_path
        self.recent_count = recent_count

    def next_run_number(self) -> int:
        """Compute the next run number based on existing files in runs/."""
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
        """Write runs/run-NNN.json and runs/run-NNN.md."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"run-{record.run_number:03d}"
        (self.runs_dir / f"{prefix}.json").write_text(
            record.to_json(), encoding="utf-8"
        )
        (self.runs_dir / f"{prefix}.md").write_text(
            record.to_markdown(), encoding="utf-8"
        )

    def generate_history(self) -> None:
        """Regenerate HISTORY.md from the runs/ directory."""
        records = self.load_records()
        if not records:
            return

        parts = [
            render_run_log(records),
            "",
            render_recent_runs(
                records, count=self.recent_count, include_steps=True
            ),
        ]
        self.history_path.write_text("\n".join(parts), encoding="utf-8")

    def load_records(self) -> list[RunRecord]:
        """Load all RunRecords sorted by run_number."""
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
    """Load the detailed execution record for a given run number on demand."""

    name = "read_run_detail"
    description = (
        "Read the detailed execution record for a specified run number, "
        "including full code and output. Use this when you need to review "
        "the details of an earlier run."
    )
    inputs = {
        "run_number": {
            "type": "integer",
            "description": "Run number (from the Run Log in Thread History)",
        },
    }
    output_type = "string"

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        super().__init__()

    def forward(self, run_number: int) -> str:
        md_path = self.runs_dir / f"run-{run_number:03d}.md"
        if not md_path.is_file():
            return f"Error: no record found for Run {run_number}."
        return md_path.read_text(encoding="utf-8")
