from __future__ import annotations

from typing import TYPE_CHECKING

from runweave.context.budget import ContextBudget

if TYPE_CHECKING:
    from smolagents.memory import MemoryStep

# 压缩后 observations 的最大字符数
_MAX_OBS_LEN = 200
# 保持完整细节的最近步数
_KEEP_RECENT = 3
# 触发压缩的阈值（占 step_budget 的比例）
_TRIGGER_RATIO = 0.8


class StepCompressor:
    """Intra-run 步骤压缩器，通过修改旧步骤字段来缩减 context。

    分三级渐进压缩（只动 ActionStep，不动 TaskStep/SystemPromptStep）：
    - Tier 1: observations 截断到 _MAX_OBS_LEN 字符
    - Tier 2: observations 清空，保留 code_action
    - Tier 3: observations = "[compressed]", code_action = None
    """

    def __init__(
        self,
        budget: ContextBudget,
        keep_recent: int = _KEEP_RECENT,
    ) -> None:
        self.budget = budget
        self.keep_recent = keep_recent
        self._threshold = int(budget.step_budget() * _TRIGGER_RATIO)

    def compress_if_needed(self, steps: list[MemoryStep]) -> None:
        """检查最近步骤的 input_tokens，超阈值则就地压缩旧步骤。

        使用最近 ActionStep 的 token_usage.input_tokens 作为实际 context 大小。
        回调触发时当前步尚未 append 到 steps，所以 steps[-1] 是上一步。
        """
        from smolagents.memory import ActionStep

        # 找到最近一个有 token_usage 的 ActionStep
        last_input_tokens = self._get_last_input_tokens(steps)
        if last_input_tokens is None or last_input_tokens < self._threshold:
            return

        # 收集可压缩的 ActionStep 索引（排除最近 keep_recent 个）
        action_indices = [
            i for i, s in enumerate(steps) if isinstance(s, ActionStep)
        ]
        if len(action_indices) <= self.keep_recent:
            return

        compressible = action_indices[: -self.keep_recent]

        # 逐级压缩：先全部 Tier 1，不够再 Tier 2，最后 Tier 3
        for tier in range(1, 4):
            for idx in compressible:
                self._apply_tier(steps[idx], tier)
            # 检查效果：重新估算（粗略，用字段长度）
            # 由于我们无法精确知道下次 input_tokens，
            # 这里一次性应用当前 tier 到所有可压缩步骤
            # 实际效果会在下一步的 token_usage 中体现
            break_after = self._estimate_reduction_sufficient(
                steps, last_input_tokens
            )
            if break_after:
                break

    def _apply_tier(self, step: MemoryStep, tier: int) -> None:
        """对单个步骤应用指定级别的压缩。"""
        from smolagents.memory import ActionStep

        if not isinstance(step, ActionStep):
            return

        if tier == 1:
            # 截断 observations
            if step.observations and len(step.observations) > _MAX_OBS_LEN:
                step.observations = step.observations[:_MAX_OBS_LEN] + "..."
        elif tier == 2:
            # 清空 observations，保留 code
            step.observations = None
            step.model_output = None
        elif tier == 3:
            # 全部清空
            step.observations = "[compressed]"
            step.code_action = None
            step.model_output = None

    @staticmethod
    def _get_last_input_tokens(steps: list[MemoryStep]) -> int | None:
        """从 steps 中找到最近一个有 token_usage 的步骤的 input_tokens。"""
        from smolagents.memory import ActionStep

        for step in reversed(steps):
            if isinstance(step, ActionStep) and step.token_usage:
                return step.token_usage.input_tokens
        return None

    @staticmethod
    def _estimate_reduction_sufficient(
        steps: list[MemoryStep],
        original_tokens: int,
    ) -> bool:
        """粗略估算压缩是否足够（基于文本长度变化）。

        这是一个近似判断：如果总文本长度减少了 30% 以上，
        认为压缩已经足够，不需要继续到更高 tier。
        """
        from smolagents.memory import ActionStep

        total_chars = 0
        for step in steps:
            if isinstance(step, ActionStep):
                if step.observations:
                    total_chars += len(step.observations)
                if step.code_action:
                    total_chars += len(step.code_action)
                if step.model_output and isinstance(step.model_output, str):
                    total_chars += len(step.model_output)

        # 粗略估算：如果剩余文本对应的 token 数在阈值以下
        estimated_tokens = total_chars / 3.5
        return estimated_tokens < original_tokens * 0.7
