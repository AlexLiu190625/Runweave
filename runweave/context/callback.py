from __future__ import annotations

from typing import Any, Callable

from runweave.context.budget import ContextBudget
from runweave.context.step_compressor import StepCompressor


def make_context_callback(budget: ContextBudget) -> Callable:
    """创建 smolagents step 回调，超阈值时压缩 memory。

    返回的回调通过 CodeAgent(step_callbacks={ActionStep: callback})
    注册，每步结束后触发。回调在 step append 到 memory 之前执行，
    所以压缩立即生效于下一步。
    """
    compressor = StepCompressor(budget)

    def _callback(step: Any, **kwargs: Any) -> None:
        agent = kwargs.get("agent")
        if agent is None:
            return
        compressor.compress_if_needed(agent.memory.steps)

    return _callback
