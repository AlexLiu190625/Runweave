from __future__ import annotations

import re

from runweave.context.budget import ContextBudget
from runweave.context.counter import TokenCounter


class InstructionCompressor:
    """将 cross-run 注入的 instructions 压缩到 token 预算以内。

    压缩策略只动 history_md（优先级最低的部分）：
    1. 去掉 Recent Runs 的 step 详情，只保留标题行
    2. 去掉 Recent Runs 整节，只留 Run Log 表
    3. Run Log 表只保留最后 10 行
    4. 硬截断 history_md 到预算剩余空间
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
        """组装并压缩 instructions，保证不超过 instruction_budget。"""
        limit = self.budget.instruction_budget()

        # 不可压缩的部分（永不裁剪）
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
            # 固定部分已超预算，丢弃 history_md
            return fixed_text or None

        # 逐级压缩 history_md
        compressed = self._compress_history(history_md, remaining)

        parts = []
        if fixed_text:
            parts.append(fixed_text)
        if compressed:
            parts.append(compressed)
        return "\n\n".join(parts) if parts else None

    def _compress_history(self, history_md: str, token_limit: int) -> str:
        """逐级压缩 HISTORY.md 直到满足 token_limit。"""
        # Level 0: 原文
        if self.counter.estimate(history_md) <= token_limit:
            return history_md

        # Level 1: 去掉 Recent Runs 的 step 详情（代码块和引用行）
        stripped = self._strip_step_details(history_md)
        if self.counter.estimate(stripped) <= token_limit:
            return stripped

        # Level 2: 去掉 Recent Runs 整节
        run_log_only = self._keep_run_log_only(history_md)
        if self.counter.estimate(run_log_only) <= token_limit:
            return run_log_only

        # Level 3: Run Log 表只保留最后 10 行
        truncated = self._truncate_run_log(run_log_only, max_rows=10)
        if self.counter.estimate(truncated) <= token_limit:
            return truncated

        # Level 4: 硬截断
        char_limit = int(token_limit * TokenCounter.CHARS_PER_TOKEN)
        return truncated[:char_limit]

    @staticmethod
    def _strip_step_details(history_md: str) -> str:
        """移除 Recent Runs 中的代码块和引用行，只保留标题。"""
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
            # 在 Recent Runs 中：保留标题和 Skills 行，跳过其余
            if line.startswith("### Run") or line.startswith("Skills:"):
                result.append(line)
            elif line.startswith("**Output:**"):
                result.append(line)
            elif line.strip() == "":
                result.append(line)

        return "\n".join(result)

    @staticmethod
    def _keep_run_log_only(history_md: str) -> str:
        """只保留 # Thread History 和 ## Run Log，丢弃 ## Recent Runs。"""
        idx = history_md.find("## Recent Runs")
        if idx == -1:
            return history_md
        return history_md[:idx].rstrip()

    @staticmethod
    def _truncate_run_log(history_md: str, max_rows: int = 10) -> str:
        """Run Log 表只保留最后 max_rows 行数据。"""
        lines = history_md.split("\n")
        # 找到表格数据行（以 | 开头，排除表头和分隔线）
        header_end = -1
        data_lines: list[int] = []
        for i, line in enumerate(lines):
            if line.startswith("|---"):
                header_end = i
            elif line.startswith("| ") and header_end >= 0:
                data_lines.append(i)

        if len(data_lines) <= max_rows:
            return history_md

        # 只保留最后 max_rows 行
        remove = set(data_lines[:-max_rows])
        return "\n".join(line for i, line in enumerate(lines) if i not in remove)
