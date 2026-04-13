"""
Example 05: Custom context budget — control token allocation.

Shows how to configure ContextBudget for different models
and budget strategies.

Usage:
    1. Copy .env.example to .env and fill in your API key / base URL
    2. python examples/05_context_budget.py
"""
from dotenv import load_dotenv
load_dotenv()

from smolagents import OpenAIServerModel
from runweave import Runtime
from runweave.context import ContextBudget

model = OpenAIServerModel(model_id="gpt-5.3")

# Custom budget: reserve more tokens for instructions (40% instead of default 25%)
budget = ContextBudget(
    model_id="gpt-5.3",
    buffer_tokens=8192,        # safety margin for output
    instruction_ratio=0.40,    # 40% of available tokens for instructions
)

print(f"Context window:      {budget.context_window:,} tokens")
print(f"Available:           {budget.available:,} tokens")
print(f"Instruction budget:  {budget.instruction_budget():,} tokens")
print(f"Step budget:         {budget.step_budget():,} tokens")
print()

rt = Runtime(model=model, context_budget=budget)

result = rt.run("Write a Python function that sorts a list using merge sort. Include comments explaining each step.")
print(f"State:  {result.state}")
print(f"Steps:  {result.step_count}")
print(f"Output: {result.output}")
