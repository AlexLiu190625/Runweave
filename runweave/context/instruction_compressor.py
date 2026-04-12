from __future__ import annotations

import re

from runweave.context.budget import ContextBudget
from runweave.context.counter import TokenCounter


class InstructionCompressor:
    """Compress cross-run injected instructions to fit within the token budget.

    Compression strategy only modifies history_md (lowest priority part):
    1. Strip step details from Recent Runs, keeping only header lines
    2. Remove the entire Recent Runs section, keeping only the Run Log table
    3. Keep only the last 10 rows of the Run Log table
    4. Hard-truncate history_md to remaining budget space
    """

    def __init__(self, budget: ContextBudget) -> None:
        self.budget = budget
        self.counter = TokenCounter()

    def compress(
        self,
        user_instructions: str | None,
        skill_catalog: str | None,
        history_md: str | None,
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

        if not history_md:
            return fixed_text or None

        remaining = limit - fixed_tokens
        if remaining <= 0:
            # Fixed parts already exceed budget, discard history_md
            return fixed_text or None

        # Progressively compress history_md
        compressed = self._compress_history(history_md, remaining)

        parts = []
        if fixed_text:
            parts.append(fixed_text)
        if compressed:
            parts.append(compressed)
        return "\n\n".join(parts) if parts else None

    def _compress_history(self, history_md: str, token_limit: int) -> str:
        """Progressively compress HISTORY.md until it fits within token_limit."""
        # Level 0: original text
        if self.counter.estimate(history_md) <= token_limit:
            return history_md

        # Level 1: strip step details from Recent Runs (code blocks and quote lines)
        stripped = self._strip_step_details(history_md)
        if self.counter.estimate(stripped) <= token_limit:
            return stripped

        # Level 2: remove the entire Recent Runs section
        run_log_only = self._keep_run_log_only(history_md)
        if self.counter.estimate(run_log_only) <= token_limit:
            return run_log_only

        # Level 3: keep only the last 10 rows of the Run Log table
        truncated = self._truncate_run_log(run_log_only, max_rows=10)
        if self.counter.estimate(truncated) <= token_limit:
            return truncated

        # Level 4: hard truncate
        char_limit = int(token_limit * TokenCounter.CHARS_PER_TOKEN)
        return truncated[:char_limit]

    @staticmethod
    def _strip_step_details(history_md: str) -> str:
        """Remove code blocks and quote lines from Recent Runs, keeping only headers."""
        lines = history_md.split("\n")
        result: list[str] = []
        in_recent = False
        in_code_block = False

        for line in lines:
            if line.startswith("## Recent Runs"):
                in_recent = True
            if not in_recent:
                result.append(line)
                continue
            # Within Recent Runs: keep headers and Skills lines, skip the rest
            if line.startswith("### Run") or line.startswith("Skills:"):
                result.append(line)
            elif line.startswith("**Output:**"):
                result.append(line)
            elif line.strip() == "":
                result.append(line)

        return "\n".join(result)

    @staticmethod
    def _keep_run_log_only(history_md: str) -> str:
        """Keep only # Thread History and ## Run Log, discard ## Recent Runs."""
        idx = history_md.find("## Recent Runs")
        if idx == -1:
            return history_md
        return history_md[:idx].rstrip()

    @staticmethod
    def _truncate_run_log(history_md: str, max_rows: int = 10) -> str:
        """Keep only the last max_rows data rows of the Run Log table."""
        lines = history_md.split("\n")
        # Find table data rows (starting with |, excluding header and separator)
        header_end = -1
        data_lines: list[int] = []
        for i, line in enumerate(lines):
            if line.startswith("|---"):
                header_end = i
            elif line.startswith("| ") and header_end >= 0:
                data_lines.append(i)

        if len(data_lines) <= max_rows:
            return history_md

        # Keep only the last max_rows rows
        remove = set(data_lines[:-max_rows])
        return "\n".join(line for i, line in enumerate(lines) if i not in remove)
