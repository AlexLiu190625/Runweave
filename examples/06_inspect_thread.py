"""
Example 06: Inspect thread — examine what Runweave persists after a run.

Runs a task, then reads back the thread's disk artifacts:
memory.json, summary.txt, HISTORY.md, and run records.

Usage:
    1. Copy .env.example to .env and fill in your API key / base URL
    2. python examples/06_inspect_thread.py
"""
from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path
from smolagents import OpenAIServerModel
from runweave import Runtime

model = OpenAIServerModel(model_id="gpt-5.3")
rt = Runtime(model=model)

result = rt.run("Create a file called hello.txt containing 'Hello, Runweave!'")
tid = result.thread_id
thread_dir = Path.home() / ".runweave" / "threads" / tid

print(f"Thread ID: {tid}")
print(f"Thread dir: {thread_dir}\n")

# 1. Workspace
workspace = thread_dir / "workspace"
print("=== Workspace ===")
for f in sorted(workspace.iterdir()):
    print(f"  {f.name}: {f.read_text().strip()[:100]}")

# 2. Summary
print("\n=== Summary ===")
summary_path = thread_dir / "summary.txt"
if summary_path.exists():
    print(summary_path.read_text().strip())

# 3. HISTORY.md
print("\n=== HISTORY.md ===")
history_path = thread_dir / "HISTORY.md"
if history_path.exists():
    print(history_path.read_text().strip()[:500])

# 4. Run records
print("\n=== Run records ===")
runs_dir = thread_dir / "runs"
for f in sorted(runs_dir.glob("*.json")):
    record = json.loads(f.read_text())
    print(f"  {f.name}: task={record['task']!r}, state={record['state']}, steps={record['step_count']}")

# 5. Memory (just show size, it's large)
print("\n=== Memory ===")
mem_path = thread_dir / "memory.json"
if mem_path.exists():
    mem = json.loads(mem_path.read_text())
    print(f"  {len(mem)} step(s) archived")
    print(f"  File size: {mem_path.stat().st_size:,} bytes")
