"""
Example 03: Custom tools — load tools from a directory.

Demonstrates ToolLoader: put .py files in a directory,
Runweave discovers and registers them automatically.

Usage:
    1. Copy .env.example to .env and fill in your API key / base URL
    2. python examples/03_custom_tools.py
"""
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from smolagents import OpenAIServerModel
from runweave import Runtime

# Prepare a tools directory with a simple tool
tools_dir = Path(__file__).parent / "_tools_demo"
tools_dir.mkdir(exist_ok=True)

(tools_dir / "word_count.py").write_text("""\
from smolagents import Tool

class WordCountTool(Tool):
    name = "word_count"
    description = "Count the number of words in a text string."
    inputs = {"text": {"type": "string", "description": "The text to count words in"}}
    output_type = "string"

    def forward(self, text: str) -> str:
        count = len(text.split())
        return f"{count} words"

word_count = WordCountTool()
""")

model = OpenAIServerModel(model_id="gpt-5.3")
rt = Runtime(model=model, tools_dir=tools_dir)

result = rt.run("Use the word_count tool to count words in: 'The quick brown fox jumps over the lazy dog'")
print(f"Output: {result.output}")
