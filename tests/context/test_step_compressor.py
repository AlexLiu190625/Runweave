from __future__ import annotations

import time

from smolagents.memory import ActionStep, TaskStep
from smolagents.monitoring import Timing, TokenUsage

from runweave.context.budget import ContextBudget
from runweave.context.step_compressor import StepCompressor


def _make_step(
    step_number: int,
    obs_len: int = 500,
    input_tokens: int = 7000,
) -> ActionStep:
    return ActionStep(
        step_number=step_number,
        timing=Timing(start_time=time.time()),
        code_action=f"result = compute_{step_number}()",
        observations="x" * obs_len,
        model_output=f"thinking about step {step_number}",
        token_usage=TokenUsage(
            input_tokens=input_tokens, output_tokens=100
        ),
    )


def test_no_compression_below_threshold():
    """token 用量低于阈值时不压缩。"""
    budget = ContextBudget("claude-sonnet-4", buffer_tokens=0)  # 200K window
    comp = StepCompressor(budget, keep_recent=2)

    steps = [_make_step(i, input_tokens=1000) for i in range(5)]
    original_obs = [s.observations for s in steps]

    comp.compress_if_needed(steps)

    # 1000 远低于 step_budget * 0.8，不应压缩
    for i, s in enumerate(steps):
        assert s.observations == original_obs[i]


def test_tier1_truncates_observations():
    """超阈值时旧步骤 observations 被截断。"""
    budget = ContextBudget("test-small", buffer_tokens=24_000)  # 8000 step budget
    comp = StepCompressor(budget, keep_recent=2)

    steps = [_make_step(i, obs_len=500, input_tokens=7000) for i in range(5)]
    comp.compress_if_needed(steps)

    # 旧步骤（0, 1, 2）应被截断
    for s in steps[:3]:
        assert len(s.observations) <= 204  # 200 + "..."

    # 最近 2 步不动
    assert len(steps[3].observations) == 500
    assert len(steps[4].observations) == 500


def test_recent_steps_always_untouched():
    """即使高压，最近 keep_recent 步也不被压缩。"""
    budget = ContextBudget("test-small", buffer_tokens=28_000)  # very tight
    comp = StepCompressor(budget, keep_recent=2)

    steps = [_make_step(i, obs_len=1000, input_tokens=3000) for i in range(5)]
    comp.compress_if_needed(steps)

    # 最近 2 步应保持原样
    assert steps[3].observations == "x" * 1000
    assert steps[4].observations == "x" * 1000
    assert steps[3].code_action is not None
    assert steps[4].code_action is not None


def test_tier3_clears_code():
    """极高压力下应升级到 Tier 3，清空 code_action。"""
    budget = ContextBudget("test-small", buffer_tokens=30_000)  # only 2000 step budget
    comp = StepCompressor(budget, keep_recent=1)

    # 每步有大量文本（长 code + 长 observations + 长 model_output）
    # 使得即使 Tier 1/2 也无法充分压缩
    steps = []
    for i in range(5):
        step = ActionStep(
            step_number=i,
            timing=Timing(start_time=time.time()),
            code_action="x = 1\n" * 200,  # 大量代码
            observations="y" * 5000,  # 大量输出
            model_output="z" * 5000,  # 大量推理
            token_usage=TokenUsage(input_tokens=1500, output_tokens=100),
        )
        steps.append(step)

    comp.compress_if_needed(steps)

    # 最老的步骤应被完全压缩（Tier 3）
    for s in steps[:4]:
        assert s.code_action is None
        assert s.observations == "[compressed]"

    # 最近 1 步不动
    assert steps[4].code_action is not None
    assert len(steps[4].observations) == 5000


def test_skips_non_action_steps():
    """TaskStep 不应被压缩。"""
    budget = ContextBudget("gpt-4", buffer_tokens=0)
    comp = StepCompressor(budget, keep_recent=1)

    task_step = TaskStep(task="Do something")
    action_steps = [_make_step(i, input_tokens=7000) for i in range(3)]
    steps = [task_step] + action_steps

    comp.compress_if_needed(steps)

    # TaskStep 保持不变
    assert steps[0].task == "Do something"


def test_no_steps_no_crash():
    """空 steps 不应报错。"""
    budget = ContextBudget("test-small")
    comp = StepCompressor(budget)
    comp.compress_if_needed([])


def test_fewer_than_keep_recent():
    """步数少于 keep_recent 时不压缩。"""
    budget = ContextBudget("gpt-4", buffer_tokens=0)
    comp = StepCompressor(budget, keep_recent=3)

    steps = [_make_step(i, input_tokens=7000) for i in range(2)]
    original_obs = [s.observations for s in steps]

    comp.compress_if_needed(steps)

    for i, s in enumerate(steps):
        assert s.observations == original_obs[i]
