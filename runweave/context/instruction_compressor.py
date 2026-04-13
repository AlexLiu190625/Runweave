from __future__ import annotations

from typing import TYPE_CHECKING

from runweave.context.budget import ContextBudget
from runweave.context.counter import TokenCounter

if TYPE_CHECKING:
    from runweave.runtime.run_record import RunRecord


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
        from runweave.runtime.run_record import render_recent_runs, render_run_log

        # Level 0: full detail
        text = render_run_log(records) + "\n\n" + render_recent_runs(records, include_steps=True)
        if self.counter.estimate(text) <= token_limit:
            return text

        # Level 1: run log + recent run headers only (no step code/output)
        text = render_run_log(records) + "\n\n" + render_recent_runs(records, include_steps=False)
        if self.counter.estimate(text) <= token_limit:
            return text

        # Level 2: run log table only
        text = render_run_log(records)
        if self.counter.estimate(text) <= token_limit:
            return text

        # Level 3: run log table, last 10 rows only
        text = render_run_log(records[-10:])
        if self.counter.estimate(text) <= token_limit:
            return text

        # Level 4: discard history entirely
        return ""
