"""
Example 09: Custom instructions — inject system-level guidance.

The `instructions` parameter is injected into the agent's system prompt.
It persists across all runs and is never compressed.

Usage:
    1. Copy .env.example to .env and fill in your API key / base URL
    2. python examples/09_custom_instructions.py
"""
from dotenv import load_dotenv
load_dotenv()

from smolagents import OpenAIServerModel
from runweave import Runtime

model = OpenAIServerModel(model_id="gpt-5.3")
rt = Runtime(
    model=model,
    instructions=(
        "You are a senior Python developer. "
        "Always follow PEP 8. "
        "Always add type hints to function signatures. "
        "Prefer f-strings over .format() or % formatting. "
        "When writing files, include a module-level docstring."
    ),
)

result = rt.run("Create a module called utils.py with functions for reading and writing JSON files.")
print(f"Output: {result.output}")

# Show what was written
from pathlib import Path
workspace = Path.home() / ".runweave" / "threads" / result.thread_id / "workspace"
utils = workspace / "utils.py"
if utils.exists():
    print(f"\n=== utils.py ===")
    print(utils.read_text())
