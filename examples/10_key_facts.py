"""
Example 10: Key Facts — curated anchor facts across runs.

Every run, Runweave produces two artifacts in parallel:

  summary.txt    — narrative summary (grows then condenses)
  key_facts.md   — curated list of stable anchor facts

The narrative is great for continuity. Key facts are the **anchor** — a short
list of goals, constraints, decisions, and produced artifacts that resist
dilution by recent activity. Both are injected into the next run's system
prompt (key_facts before the narrative), so the agent stays aligned with
the original intent even after many runs.

This example runs three related tasks on the same thread and prints the
key_facts file after each run so you can watch it evolve.

Usage:
    python examples/10_key_facts.py
"""
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

from smolagents import OpenAIServerModel
from runweave import Runtime

model = OpenAIServerModel(model_id="gpt-5.3")
rt = Runtime(model=model)


def _print_key_facts(thread_id: str, header: str) -> None:
    kf = Path.home() / ".runweave" / "threads" / thread_id / "key_facts.md"
    print(f"\n--- {header} ---")
    if kf.is_file():
        print(kf.read_text())
    else:
        print("(not yet created)")
    print()


# Run 1 — establish the goal and an early constraint
print("=== Run 1 ===")
r1 = rt.run(
    "Build a CLI tool todo.py that stores tasks in a JSON file at .todo.json. "
    "Do NOT use any third-party libraries."
)
print(f"Thread: {r1.thread_id}\nState:  {r1.state}\n")
_print_key_facts(r1.thread_id, "key_facts after run 1")

# Run 2 — add a feature; the distiller should retain the "no third-party" constraint
print("=== Run 2 ===")
r2 = rt.run(
    "Add a `done <id>` subcommand that marks a task as completed.",
    thread_id=r1.thread_id,
)
print(f"State:  {r2.state}\n")
_print_key_facts(r2.thread_id, "key_facts after run 2")

# Run 3 — a decision that should be recorded
print("=== Run 3 ===")
r3 = rt.run(
    "Add a `list` subcommand. If the terminal supports color, use ANSI escapes; "
    "otherwise print plain text.",
    thread_id=r1.thread_id,
)
print(f"State:  {r3.state}\n")
_print_key_facts(r3.thread_id, "key_facts after run 3")

print("=== Summary (narrative track, for comparison) ===")
summary = Path.home() / ".runweave" / "threads" / r1.thread_id / "summary.txt"
print(summary.read_text() if summary.is_file() else "(none)")
