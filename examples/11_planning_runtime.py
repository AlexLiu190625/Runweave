"""
Example 11: PlanningRuntime — planner-driven multi-step execution with model routing.

This example shows the v0.3 PlanningRuntime in action. A single user task is:
  1. Decomposed by a planner LLM into a list of steps (each with metadata)
  2. Each step is routed to the best-fit model by the Router
  3. Each step is executed by a smolagents.CodeAgent on the chosen model
  4. The final aggregated output is returned as a single RunResult

After the run, inspect the archived plan in:
    ~/.runweave/threads/<thread-id>/plans/plan-001.json

to see which model was selected for each step (`selected_model_id`) and the
deterministic execution trace.

Usage:
    python examples/11_planning_runtime.py
"""
from dotenv import load_dotenv
load_dotenv()

from smolagents import OpenAIServerModel

from runweave import ModelProfile, PlanningRuntime, Router

# Construct concrete smolagents Model instances for each candidate.
# Substitute your own model_id strings as needed.
haiku = OpenAIServerModel(model_id="claude-haiku-4-5-20251001")
sonnet = OpenAIServerModel(model_id="claude-sonnet-4-6")
opus = OpenAIServerModel(model_id="claude-opus-4-7")

# ModelProfile wraps a Model with capability metadata used by Router.
# coding_score / long_context_score are hand-labeled 0.0-1.0 estimates.
models = [
    ModelProfile(
        model=haiku,
        context_window=200_000,
        supports_tools=True,
        supports_structured_output=True,
        coding_score=0.7,
        long_context_score=0.65,
        latency="low",
        cost_tier="low",
    ),
    ModelProfile(
        model=sonnet,
        context_window=1_000_000,
        supports_tools=True,
        supports_structured_output=True,
        coding_score=0.9,
        long_context_score=0.85,
        latency="medium",
        cost_tier="medium",
    ),
    ModelProfile(
        model=opus,
        context_window=1_000_000,
        supports_tools=True,
        supports_structured_output=True,
        coding_score=0.95,
        long_context_score=0.9,
        latency="high",
        cost_tier="high",
    ),
]

# The planner is typically the strongest model — it sees only the task and
# thread context (key_facts + summary) and outputs JSON.
rt = PlanningRuntime(
    planner_model=opus,
    models=models,
    router=Router(),
)

result = rt.run(
    "Generate a Python dataclass `Task` with fields id/title/done, write unit "
    "tests for it, then write a one-paragraph README explaining how to use it."
)

print(f"\nThread: {result.thread_id}")
print(f"State:  {result.state}")
print(f"Steps:  {result.step_count}")
print(f"\nOutput:\n{result.output}\n")
print(
    f"Inspect the archived plan at:\n"
    f"  ~/.runweave/threads/{result.thread_id}/plans/plan-001.json\n"
    f"to see which model was selected for each step."
)
