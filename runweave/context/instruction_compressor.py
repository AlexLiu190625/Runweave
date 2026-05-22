from __future__ import annotations

from runweave.context.budget import ContextBudget
from runweave.context.counter import TokenCounter
from runweave.runtime.history import READ_RUN_DETAIL_TOOL_NAME
from runweave.runtime.run_record import (
    DetailLevel,
    RunRecord,
    render_run_log,
)


# Boundaries for U-shaped middle bucketing (fraction of distance from tail).
# Heuristic; revisit after collecting real usage telemetry.
_THIRD = 1 / 3
_TWO_THIRDS = 2 / 3


def _assign_levels(
    n: int, head_count: int, tail_count: int
) -> list[DetailLevel]:
    """Assign a base DetailLevel to each run index 0..n-1 (no budget pressure).

    Head runs and tail runs are FULL. Middle runs are bucketed by distance from
    tail: closer to tail = more detail (TAKEAWAY → TITLE → LOG_LINE).
    """
    levels = [DetailLevel.FULL] * n
    if n <= head_count + tail_count:
        return levels

    middle_start = head_count
    middle_end = n - tail_count
    middle_len = middle_end - middle_start

    for i in range(middle_start, middle_end):
        dist_from_tail = middle_end - i        # 1..middle_len; smaller = nearer tail
        frac = (dist_from_tail - 1) / middle_len
        if frac < _THIRD:
            levels[i] = DetailLevel.TAKEAWAY
        elif frac < _TWO_THIRDS:
            levels[i] = DetailLevel.TITLE
        else:
            levels[i] = DetailLevel.LOG_LINE
    return levels


def _degrade(
    levels: list[DetailLevel],
    budget_level: int,
    head_count: int,
    tail_count: int,
) -> list[DetailLevel]:
    """Apply budget pressure to a level vector. Head/tail indices untouched."""
    n = len(levels)
    out = list(levels)
    if budget_level == 0:
        return out

    middle_range = range(head_count, max(n - tail_count, head_count))
    if budget_level == 1:
        for i in middle_range:
            out[i] = DetailLevel(max(int(out[i]) - 1, 0))
        return out
    # budget_level >= 2: middle collapsed to LOG_LINE.
    for i in middle_range:
        out[i] = DetailLevel.LOG_LINE
    return out


def _render_history(
    records: list[RunRecord],
    levels: list[DetailLevel],
    budget_level: int,
    head_count: int,
    tail_count: int,
) -> str:
    """Compose the full history block at the given budget level.

    Levels 0-2: Run Log over all records + Recent Runs section honoring ``levels``.
    Level 3: Run Log over head+tail only + Recent Runs section for head+tail FULL;
             omission hint if middle is non-empty.
    Level 4: Run Log over head+tail only, no Recent Runs section.

    Head and tail are always rendered FULL regardless of ``levels`` content.
    """
    n = len(records)
    head = records[:head_count]
    # Cap tail start so it never overlaps with head.
    tail = records[max(n - tail_count, head_count):]
    middle = (
        records[head_count : n - tail_count]
        if n > head_count + tail_count
        else []
    )

    if budget_level >= 3:
        table_records = head + tail
        omitted = len(middle)
    else:
        # TODO: cap table rows for very long threads (N >> 100); the Run Log
        # itself grows linearly with N and is currently never truncated.
        table_records = list(records)
        omitted = 0

    parts: list[str] = [render_run_log(table_records)]

    if budget_level == 4:
        return "\n".join(parts)

    parts.extend(["", "## Recent Runs", ""])

    # Newest-first ordering: tail (newest→oldest) → middle (newest→oldest) → head.
    section_records = (
        list(reversed(tail))
        + list(reversed(middle))
        + list(reversed(head))
    )
    middle_levels_rev = [
        levels[i] for i in reversed(range(head_count, n - tail_count))
    ]
    section_levels = (
        [DetailLevel.FULL] * len(tail)
        + middle_levels_rev
        + [DetailLevel.FULL] * len(head)
    )

    for r, lvl in zip(section_records, section_levels):
        if lvl == DetailLevel.LOG_LINE:
            continue
        block = r.render_at_level(lvl)
        if block:
            parts.append(block)

    if budget_level == 3 and omitted > 0:
        parts.append(
            f"\n*({omitted} middle runs omitted; "
            f"use {READ_RUN_DETAIL_TOOL_NAME}(N) to fetch any of them.)*"
        )

    return "\n".join(parts)


class InstructionCompressor:
    """Compress cross-run injected instructions to fit within the token budget.

    Works on structured RunRecord data rather than parsing markdown text.
    Compression only reduces history detail; fixed parts (user_instructions,
    skill_catalog, key_facts, thread_summary) are never trimmed.

    History compression preserves head and tail runs FULL and progressively
    compresses middle runs (U-shaped decay). The five budget levels are:

    L0: head FULL + middle U-shape + tail FULL  (full detail)
    L1: middle bumped down one notch; head/tail FULL
    L2: middle all LOG_LINE (table-only); head/tail FULL
    L3: middle dropped from Run Log table too; head/tail FULL section + omission hint
    L4: no Recent Runs section; Run Log over head+tail only
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
        key_facts: str | None = None,
    ) -> str | None:
        """Assemble and compress instructions to stay within instruction_budget.

        Fixed parts are never trimmed. Ordering:
        user_instructions -> skill_catalog -> key_facts -> thread_summary.
        key_facts is a curated anchor list that complements the narrative
        summary — it deserves the same non-compressible guarantee.
        """
        limit = self.budget.instruction_budget()

        # Non-compressible parts (never trimmed)
        fixed_parts: list[str] = []
        if user_instructions:
            fixed_parts.append(user_instructions)
        if skill_catalog:
            fixed_parts.append(skill_catalog)
        if key_facts:
            fixed_parts.append(f"\n## Key Facts\n{key_facts}")
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
        """Render history at the most detailed budget level that fits."""
        head_count = self.budget.head_count
        tail_count = self.budget.tail_count
        base_levels = _assign_levels(len(records), head_count, tail_count)
        for budget_level in range(5):
            levels = _degrade(base_levels, budget_level, head_count, tail_count)
            text = _render_history(
                records, levels, budget_level, head_count, tail_count
            )
            if self.counter.estimate(text) <= token_limit:
                return text
        return ""
