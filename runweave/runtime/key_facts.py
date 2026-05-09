from __future__ import annotations

from typing import Any, TYPE_CHECKING

from smolagents.models import ChatMessage, MessageRole

if TYPE_CHECKING:
    from smolagents.models import Model

# System prompt for key-facts distillation. The distiller maintains a stable,
# curated list of high-value facts that anchor future runs of the thread —
# complementary to, not a replacement for, the narrative summary.
_SYSTEM_PROMPT = (
    "You maintain a Key Facts list for an agent thread: a curated set of "
    "stable, high-value facts that anchor future runs.\n"
    "\n"
    "KEEP facts that are:\n"
    "  - goals and hard constraints from the user\n"
    "  - selected approaches or rejected alternatives (with reason)\n"
    "  - verified invariants, schemas, contracts\n"
    "  - produced artifacts (filenames, endpoints, versions)\n"
    "  - current state of long-running work\n"
    "\n"
    "DROP:\n"
    "  - exploration dead-ends and debug trails\n"
    "  - tool-call boilerplate\n"
    "  - details already stored in workspace files\n"
    "  - ephemeral intermediate values\n"
    "\n"
    "Rules:\n"
    "  - Each fact is one bullet line prefixed with [run N]\n"
    "  - If a new fact supersedes or contradicts an existing one, REPLACE "
    "    the old entry (use the newer [run N] tag). Do not append both.\n"
    "  - If an existing fact is still valid, keep it verbatim.\n"
    "  - Keep total <= 30 entries; drop the least load-bearing if over.\n"
    "  - Output ONLY the bullet list. No header, no commentary, no code "
    "    fences."
)

# Soft cap on how much of the run output to feed into the distiller prompt.
# The distiller only needs enough to identify facts; workspace artifacts live
# on disk and are referenced by filename.
_OUTPUT_TRUNCATE_CHARS = 4000


class KeyFactsDistiller:
    """Produce or update a thread's Key Facts list via an LLM call.

    Runs as a second, independent LLM call alongside SummaryGenerator.
    Unlike the narrative summary (which grows then condenses), Key Facts
    is a curated anchor: stable entries that survive across many runs and
    resist dilution by recent activity.
    """

    def __init__(self, model: Model) -> None:
        self.model = model

    def distill(
        self,
        task: str,
        output: Any,
        run_number: int,
        previous_key_facts: str | None = None,
    ) -> str:
        """Return the updated Key Facts bullet list as plain text.

        The returned string contains only bullets — no Markdown header —
        matching the storage convention used by ``summary.txt``.
        """
        parts: list[str] = []
        if previous_key_facts and previous_key_facts.strip():
            parts.append(f"## Current Key Facts\n{previous_key_facts.strip()}")
        else:
            parts.append("## Current Key Facts\n(empty — this is the first distillation)")

        output_str = str(output) if output is not None else ""
        if len(output_str) > _OUTPUT_TRUNCATE_CHARS:
            output_str = output_str[:_OUTPUT_TRUNCATE_CHARS] + "...[truncated]"

        parts.append(
            f"## This Run (run {run_number})\n"
            f"**Task:** {task}\n"
            f"**Output:** {output_str}"
        )
        parts.append(
            "Return the updated Key Facts bullet list. "
            "Apply the rules from the system prompt."
        )

        user_content = "\n\n".join(parts)

        response = self.model(
            messages=[
                ChatMessage(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
                ChatMessage(role=MessageRole.USER, content=user_content),
            ]
        )

        return _normalize(response.content)


def _normalize(text: str) -> str:
    """Strip a stray ``# Key Facts`` header or code fences if the model added one."""
    stripped = text.strip()
    # Remove code fences if the model wrapped the list
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if len(lines) >= 2:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
    # Remove a stray H1 header
    if stripped.lower().startswith("# key facts"):
        _, _, rest = stripped.partition("\n")
        stripped = rest.strip()
    return stripped
