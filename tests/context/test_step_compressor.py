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
    """No compression when token usage is below threshold."""
    budget = ContextBudget("claude-sonnet-4", buffer_tokens=0)  # 200K window
    comp = StepCompressor(budget, keep_recent=2)

    steps = [_make_step(i, input_tokens=1000) for i in range(5)]
    original_obs = [s.observations for s in steps]

    comp.compress_if_needed(steps)

    # 1000 is well below step_budget * 0.8, no compression expected
    for i, s in enumerate(steps):
        assert s.observations == original_obs[i]


def test_tier1_truncates_observations():
    """Observations on old steps should be truncated when above threshold."""
    budget = ContextBudget("test-small", buffer_tokens=24_000)  # 8000 step budget
    comp = StepCompressor(budget, keep_recent=2)

    steps = [_make_step(i, obs_len=500, input_tokens=7000) for i in range(5)]
    comp.compress_if_needed(steps)

    # Old steps (0, 1, 2) should be truncated
    for s in steps[:3]:
        assert len(s.observations) <= 204  # 200 + "..."

    # Most recent 2 steps untouched
    assert len(steps[3].observations) == 500
    assert len(steps[4].observations) == 500


def test_recent_steps_always_untouched():
    """Most recent keep_recent steps should never be compressed, even under high pressure."""
    budget = ContextBudget("test-small", buffer_tokens=28_000)  # very tight
    comp = StepCompressor(budget, keep_recent=2)

    steps = [_make_step(i, obs_len=1000, input_tokens=3000) for i in range(5)]
    comp.compress_if_needed(steps)

    # Most recent 2 steps should remain intact
    assert steps[3].observations == "x" * 1000
    assert steps[4].observations == "x" * 1000
    assert steps[3].code_action is not None
    assert steps[4].code_action is not None


def test_tier3_clears_code():
    """Under extreme pressure, should escalate to Tier 3 and clear code_action."""
    budget = ContextBudget("test-small", buffer_tokens=30_000)  # only 2000 step budget
    comp = StepCompressor(budget, keep_recent=1)

    # Each step has large text (long code + observations + model_output)
    # so that even Tier 1/2 cannot compress sufficiently
    steps = []
    for i in range(5):
        step = ActionStep(
            step_number=i,
            timing=Timing(start_time=time.time()),
            code_action="x = 1\n" * 200,  # large code block
            observations="y" * 5000,  # large output
            model_output="z" * 5000,  # large reasoning
            token_usage=TokenUsage(input_tokens=1500, output_tokens=100),
        )
        steps.append(step)

    comp.compress_if_needed(steps)

    # Oldest steps should be fully compressed (Tier 3)
    for s in steps[:4]:
        assert s.code_action is None
        assert s.observations == "[compressed]"

    # Most recent 1 step untouched
    assert steps[4].code_action is not None
    assert len(steps[4].observations) == 5000


def test_skips_non_action_steps():
    """TaskStep should not be compressed."""
    budget = ContextBudget("gpt-4", buffer_tokens=0)
    comp = StepCompressor(budget, keep_recent=1)

    task_step = TaskStep(task="Do something")
    action_steps = [_make_step(i, input_tokens=7000) for i in range(3)]
    steps = [task_step] + action_steps

    comp.compress_if_needed(steps)

    # TaskStep remains unchanged
    assert steps[0].task == "Do something"


def test_no_steps_no_crash():
    """Empty steps list should not cause errors."""
    budget = ContextBudget("test-small")
    comp = StepCompressor(budget)
    comp.compress_if_needed([])


def test_fewer_than_keep_recent():
    """No compression when step count is less than keep_recent."""
    budget = ContextBudget("gpt-4", buffer_tokens=0)
    comp = StepCompressor(budget, keep_recent=3)

    steps = [_make_step(i, input_tokens=7000) for i in range(2)]
    original_obs = [s.observations for s in steps]

    comp.compress_if_needed(steps)

    for i, s in enumerate(steps):
        assert s.observations == original_obs[i]
