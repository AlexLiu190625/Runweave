"""
Example 02: Resume — two runs on the same thread.

Run 1 creates a script. Run 2 continues in the same workspace,
receiving a summary of Run 1 instead of the full memory.

Usage:
    1. Copy .env.example to .env and fill in your API key / base URL
    2. python examples/02_resume_thread.py
"""
from dotenv import load_dotenv
load_dotenv()

from smolagents import OpenAIServerModel
from runweave import Runtime

model = OpenAIServerModel(model_id="gpt-5.3")
rt = Runtime(model=model)

# Run 1
print("=== Run 1 ===")
r1 = rt.run("Write a Python script called calc.py that evaluates arithmetic expressions from stdin.")
print(f"Thread: {r1.thread_id}")
print(f"State:  {r1.state}")
print(f"Output: {r1.output}\n")

# Run 2 — same thread
print("=== Run 2 ===")
r2 = rt.run(
    "Add support for parentheses and write unit tests in test_calc.py.",
    thread_id=r1.thread_id,
)
print(f"State:  {r2.state}")
print(f"Output: {r2.output}\n")

# Show what's in the workspace
from pathlib import Path
workspace = Path.home() / ".runweave" / "threads" / r1.thread_id / "workspace"
print("=== Workspace contents ===")
for f in sorted(workspace.iterdir()):
    print(f"  {f.name}")
