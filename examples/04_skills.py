"""
Example 04: Skill system — load reusable instruction documents.

Creates a demo skill directory, then runs a task that triggers
the agent to load and follow the skill instructions.

Usage:
    1. Copy .env.example to .env and fill in your API key / base URL
    2. python examples/04_skills.py
"""
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from smolagents import OpenAIServerModel
from runweave import Runtime

# Prepare a skills directory with a "code_review" skill
skills_dir = Path(__file__).parent / "_skills_demo"
skill_path = skills_dir / "code_review"
skill_path.mkdir(parents=True, exist_ok=True)

(skill_path / "SKILL.md").write_text("""\
---
name: "code_review"
description: "Review Python code for common issues and suggest improvements"
---

# Code Review

When reviewing Python code, check for:

1. Missing type hints on public functions
2. Bare except clauses (should catch specific exceptions)
3. Mutable default arguments
4. Missing docstrings on public classes/functions
5. Unused imports

Format your review as a numbered list of findings.
Each finding should include the line reference and a suggested fix.
""")

# Put a reference file for the skill
refs = skill_path / "references"
refs.mkdir(exist_ok=True)
(refs / "style_guide.md").write_text("""\
# Style Guide

- Use snake_case for functions and variables
- Use PascalCase for classes
- Maximum line length: 88 characters (black default)
- Prefer pathlib over os.path
""")

model = OpenAIServerModel(model_id="gpt-5.3")
rt = Runtime(
    model=model,
    skills_dir=skills_dir,
)

code_to_review = '''
def process(data, cache={}):
    try:
        for item in data:
            cache[item.id] = item.value
        return cache
    except:
        return None
'''

result = rt.run(f"Review this Python code using the code_review skill:\n```python\n{code_to_review}\n```")
print(f"Skills used: {result.skills_used}")
print(f"Review:\n{result.output}")
