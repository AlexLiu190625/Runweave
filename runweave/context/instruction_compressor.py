from __future__ import annotations

from typing import TYPE_CHECKING

from runweave.context.budget import ContextBudget
from runweave.context.counter import TokenCounter

if TYPE_CHECKING:
    from runweave.runtime.run_record import RunRecord


# Max code lines per step in rendered history
_MAX_CODE_LINES = 10
# Default number of recent runs to show in detail
_RECENT_COUNT = 3


class InstructionCompressor:
    """Compress cross-run injected instructions to fit within the token budget.

    Works on structured RunRecord data rather than parsing markdown text.
    Compression strategy only reduces history detail (lowest priority part):

    Level 0: full history — run log table + recent runs with code and output
    Level 1: strip step details from recent runs (remove code blocks and output)
    Level 2: run log table only, no recent runs section
    Level 3: truncate run log to last 10 rows
    Level 4: discard history entirely
    """

    def __init__(self, budget: ContextBudget) -> None:
        self.budget = budget
        self.counter = TokenCounter()

    def compress(
        self,
        user_instructions: str | None,
        skill_catalog: str | None,
        history_records: list[RunRecord] | None,
        thread_summary: str | None,
    ) -> str | None:
        """Assemble and compress instructions to stay within instruction_budget."""
        limit = self.budget.instruction_budget()

        # Non-compressible parts (never trimmed)
        fixed_parts: list[str] = []
        if user_instructions:
            fixed_parts.append(user_instructions)
        if skill_catalog:
            fixed_parts.append(skill_catalog)
        if thread_summary:
            fixed_parts.append(f"\n## Thread Summary\n{thread_summary}")

        fixed_text = "\n\n".join(fixed_parts) if fixed_parts else ""
        fixed_tokens = self.counter.estimate(fixed_text)

        if not history_records:
            return fixed_text or None

        remaining = limit - fixed_tokens
        if remaining <= 0:
            # Fixed parts already exceed budget; discard history
            return fixed_text or None

        # Progressively render history at decreasing detail levels
        history_text = self._render_within_budget(history_records, remaining)

        parts = []
        if fixed_text:
            parts.append(fixed_text)
        if history_text:
            parts.append(history_text)
        return "\n\n".join(parts) if parts else None

    def _render_within_budget(
        self, records: list[RunRecord], token_limit: int
    ) -> str:
        """Render history at the most detailed level that fits the budget."""
        # Level 0: full detail
        text = self._render_full(records)
        if self.counter.estimate(text) <= token_limit:
            return text

        # Level 1: run log + recent run headers only (no step code/output)
        text = self._render_headers_only(records)
        if self.counter.estimate(text) <= token_limit:
            return text

        # Level 2: run log table only
        text = self._render_run_log(records)
        if self.counter.estimate(text) <= token_limit:
            return text

        # Level 3: run log table, last 10 rows only
        text = self._render_run_log(records[-10:])
        if self.counter.estimate(text) <= token_limit:
            return text

        # Level 4: discard history entirely
        return ""

    # -- Rendering methods ------------------------------------------------

    @staticmethod
    def _render_run_log(records: list[RunRecord]) -> str:
        """Render just the Run Log table."""
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
            task_safe = task_short.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {r.run_number} | {time_short} "
                f"| {task_safe} | {r.state} | {skills} | {tools} |"
            )
        return "\n".join(lines)

    @classmethod
    def _render_headers_only(cls, records: list[RunRecord]) -> str:
        """Render run log + recent run headers without step details."""
        lines = [cls._render_run_log(records), ""]
        recent = records[-_RECENT_COUNT:]
        lines.append("## Recent Runs")
        lines.append("")
        for r in reversed(recent):
            skills_str = ", ".join(r.skills_used) if r.skills_used else "—"
            tools_str = ", ".join(r.tools_used) if r.tools_used else "—"
            task_safe = r.task[:50].replace("\n", " ")
            lines.append(f"### Run {r.run_number} — {task_safe} ({r.state})")
            lines.append(f"Skills: {skills_str} | Tools: {tools_str} | Steps: {r.step_count}")
            output_safe = str(r.output)[:200].replace("\n", " ")
            lines.append(f"**Output:** {output_safe}")
            lines.append("")
        return "\n".join(lines)

    @classmethod
    def _render_full(cls, records: list[RunRecord]) -> str:
        """Render run log + recent runs with full step details."""
        lines = [cls._render_run_log(records), ""]
        recent = records[-_RECENT_COUNT:]
        lines.append("## Recent Runs")
        lines.append("")
        for r in reversed(recent):
            skills_str = ", ".join(r.skills_used) if r.skills_used else "—"
            tools_str = ", ".join(r.tools_used) if r.tools_used else "—"
            task_safe = r.task[:50].replace("\n", " ")
            lines.append(f"### Run {r.run_number} — {task_safe} ({r.state})")
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
                    output_text = step.output[:500]
                    if len(step.output) > 500:
                        output_text += "..."
                    for quote_line in output_text.split("\n"):
                        lines.append(f"> {quote_line}")
                lines.append("")
            output_safe = str(r.output)[:200].replace("\n", " ")
            lines.append(f"**Output:** {output_safe}")
            lines.append("")
        return "\n".join(lines)
