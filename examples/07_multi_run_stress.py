"""
Example 07: Multi-run stress test — verify context doesn't overflow.

Runs 5 consecutive tasks on the same thread to verify that:
- Summaries accumulate correctly
- History grows without errors
- Context compression kicks in on long threads

Usage:
    1. Copy .env.example to .env and fill in your API key / base URL
    2. python examples/07_multi_run_stress.py
"""
from dotenv import load_dotenv
load_dotenv()

from smolagents import OpenAIServerModel
from runweave import Runtime

model = OpenAIServerModel(model_id="gpt-5.3")
rt = Runtime(model=model)

tasks = [
    "Create a file called notes.md with a title '# Project Notes'.",
    "Add a section '## Day 1' to notes.md with a paragraph about getting started.",
    "Add a section '## Day 2' to notes.md about implementing the core feature.",
    "Add a section '## Day 3' about writing tests and fixing bugs.",
    "Read notes.md and write a one-paragraph summary at the top of the file.",
]

thread_id = None

for i, task in enumerate(tasks, 1):
    print(f"=== Run {i}/{len(tasks)} ===")
    print(f"Task: {task}")
    result = rt.run(task, thread_id=thread_id)
    thread_id = result.thread_id
    print(f"State: {result.state}")
    print(f"Steps: {result.step_count}")
    print(f"Summary length: {len(result.summary)} chars")
    print()

print(f"Thread ID: {thread_id}")
print(f"All {len(tasks)} runs completed on the same thread.")

# Show final workspace
from pathlib import Path
workspace = Path.home() / ".runweave" / "threads" / thread_id / "workspace"
print(f"\nFinal workspace contents:")
for f in sorted(workspace.iterdir()):
    print(f"  {f.name} ({f.stat().st_size} bytes)")
