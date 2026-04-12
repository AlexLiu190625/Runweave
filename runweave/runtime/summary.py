from __future__ import annotations

from typing import Any, TYPE_CHECKING

from smolagents.models import ChatMessage, MessageRole

if TYPE_CHECKING:
    from smolagents.models import Model

# System prompt for summary generation
_SYSTEM_PROMPT = (
    "You are a summary assistant. Your task is to maintain a summary of an "
    "agent's work thread, recording what the agent accomplished across multiple "
    "runs. The summary should be concise (under 200-300 words), focusing on: "
    "what tasks were completed, what files or results were produced, and the "
    "current state of work. Output only the summary itself with no extra commentary."
)


class SummaryGenerator:
    """Generate or update a thread summary using an LLM."""

    def __init__(self, model: Model) -> None:
        self.model = model

    def generate(
        self,
        task: str,
        output: Any,
        previous_summary: str | None = None,
    ) -> str:
        """Generate or update the thread summary based on the latest run.

        When previous_summary is provided, appends new information to it;
        otherwise generates an initial summary. The summary accumulates
        information across all runs.
        """
        # Build user prompt
        parts: list[str] = []
        if previous_summary:
            parts.append(f"## Current Summary\n{previous_summary}")
            parts.append("Please update the above summary by appending information from this run.")
        else:
            parts.append("This is the first run of this thread. Please generate an initial summary.")

        parts.append(f"## Task\n{task}")
        parts.append(f"## Output\n{output}")

        user_content = "\n\n".join(parts)

        # Call LLM
        response = self.model(messages=[
            ChatMessage(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_content),
        ])

        return response.content.strip()
