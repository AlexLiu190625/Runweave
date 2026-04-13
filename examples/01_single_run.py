"""
Example 01: Single run — the simplest possible Runweave usage.

Creates a new thread, runs one task, and prints the result.
A thread and workspace are created automatically under ~/.runweave/.

Usage:
    1. Copy .env.example to .env and fill in your API key / base URL
    2. python examples/01_single_run.py
"""
from dotenv import load_dotenv
load_dotenv()

from smolagents import OpenAIServerModel
from runweave import Runtime

model = OpenAIServerModel(model_id="gpt-5.3")
rt = Runtime(model=model)

result = rt.run("Write a Python function that checks if a number is prime.")

print(f"Thread:  {result.thread_id}")
print(f"State:   {result.state}")
print(f"Steps:   {result.step_count}")
print(f"Output:  {result.output}")
print(f"Summary: {result.summary}")
