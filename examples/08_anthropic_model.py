"""
Example 08: Using Anthropic models instead of OpenAI.

smolagents supports multiple providers. This example shows
how to use Claude with Runweave.

Usage:
    1. Set ANTHROPIC_API_KEY in .env or export it
    2. python examples/08_anthropic_model.py
"""
from dotenv import load_dotenv
load_dotenv()

from smolagents import InferenceClientModel
from runweave import Runtime

# Use Anthropic via smolagents' InferenceClientModel
# Requires: pip install huggingface_hub
model = InferenceClientModel(model_id="claude-sonnet-4-20250514")

rt = Runtime(model=model)

result = rt.run("Write a haiku about programming.")
print(f"Output: {result.output}")
print(f"Thread: {result.thread_id}")
