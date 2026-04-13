from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from smolagents import CodeAgent, Tool
from smolagents.memory import ActionStep
from smolagents.models import Model

from runweave.runtime.run_record import _count_action_steps

from runweave.context import (
    ContextBudget,
    InstructionCompressor,
    make_context_callback,
)
from runweave.executor.workspace_executor import WorkspaceExecutor
from runweave.runtime.history import HistoryWriter, ReadRunDetailTool
from runweave.runtime.memory_io import save_memory
from runweave.runtime.result import RunResult
from runweave.runtime.run_record import extract_run_record
from runweave.runtime.summary import SummaryGenerator
from runweave.runtime.thread import Thread
from runweave.runtime.thread_store import ThreadStore
from runweave.skill.loader import SkillLoader
from runweave.tool.loader import ToolLoader

# Default data directory
DEFAULT_BASE_DIR = Path.home() / ".runweave"


def _build_result_from_memory(agent: CodeAgent) -> Any:
    """Build a smolagents-RunResult-compatible object from agent memory.

    After ``agent.run(stream=True)`` is fully consumed, smolagents does
    not build a RunResult.  This function mirrors that post-stream logic
    so ``_finalize_run`` receives the same object shape.
    """
    import types

    from smolagents.agents import AgentMaxStepsError, FinalAnswerStep
    from smolagents.memory import PlanningStep

    # Output — last FinalAnswerStep
    output = None
    for step in reversed(agent.memory.steps):
        if isinstance(step, FinalAnswerStep):
            output = step.output
            break

    # State
    if agent.memory.steps and isinstance(
        getattr(agent.memory.steps[-1], "error", None), AgentMaxStepsError
    ):
        state = "max_steps_error"
    else:
        state = "success"

    # Token usage
    total_input = 0
    total_output = 0
    has_usage = True
    for step in agent.memory.steps:
        if isinstance(step, (ActionStep, PlanningStep)):
            if step.token_usage is None:
                has_usage = False
                break
            total_input += step.token_usage.input_tokens
            total_output += step.token_usage.output_tokens

    token_usage = (
        types.SimpleNamespace(
            dict=lambda: {"input_tokens": total_input, "output_tokens": total_output}
        )
        if has_usage
        else None
    )

    return types.SimpleNamespace(
        output=output,
        state=state,
        steps=agent.memory.steps,
        token_usage=token_usage,
        timing=None,
    )


class Runtime:
    """Top-level entry point for Runweave.

    Wraps a smolagents CodeAgent in a persistent thread: each run gets
    its own workspace and memory archive.
    """

    def __init__(
        self,
        model: Model,
        tools: list[Tool] | None = None,
        instructions: str | None = None,
        base_dir: Path | None = None,
        additional_authorized_imports: list[str] | None = None,
        skills_dir: Path | None = None,
        tools_dir: Path | None = None,
        context_budget: ContextBudget | None = None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
    ) -> None:
        self.model = model
        self.tools = tools or []
        self.instructions = instructions
        self.additional_authorized_imports = additional_authorized_imports or []
        self.store = ThreadStore(base_dir or DEFAULT_BASE_DIR)
        self.skill_loader = SkillLoader(skills_dir) if skills_dir else None
        self.tool_loader = ToolLoader(tools_dir) if tools_dir else None
        self.context_budget = context_budget or ContextBudget(model.model_id)
        self.planning_interval = planning_interval
        self.stream_outputs = stream_outputs

    def run(
        self,
        task: str,
        thread_id: str | None = None,
        tool_names: list[str] | None = None,
    ) -> RunResult:
        """Execute a task within the specified thread and return a RunResult.

        When thread_id is None a new thread is created automatically;
        passing an existing thread_id continues work in the same workspace.
        """
        agent, thread, tools_used = self._prepare_run(thread_id, tool_names)
        smolagents_result = agent.run(task, return_full_result=True)
        return self._finalize_run(agent, thread, task, smolagents_result, tools_used)

    def run_stream(
        self,
        task: str,
        thread_id: str | None = None,
        tool_names: list[str] | None = None,
    ) -> Generator[Any, None, RunResult]:
        """Execute a task in streaming mode.

        Yields step events (ActionStep, PlanningStep, ChatMessageStreamDelta,
        etc.) as the agent executes.  After the generator is fully consumed,
        its return value is a RunResult (accessible via StopIteration.value).
        """
        agent, thread, tools_used = self._prepare_run(thread_id, tool_names)

        for event in agent.run(task, stream=True):
            yield event

        smolagents_result = _build_result_from_memory(agent)
        return self._finalize_run(agent, thread, task, smolagents_result, tools_used)

    def _prepare_run(
        self,
        thread_id: str | None,
        tool_names: list[str] | None,
    ) -> tuple[CodeAgent, Thread, list[str]]:
        """Set up thread, executor, tools, and agent for a run."""
        # 1. Load or create the thread
        if thread_id and self.store.exists(thread_id):
            thread = self.store.load(thread_id)
        else:
            thread = self.store.create(thread_id)

        # 2. Build the WorkspaceExecutor
        executor = WorkspaceExecutor(
            workspace_dir=thread.workspace_dir,
            additional_authorized_imports=self.additional_authorized_imports,
        )

        # 3. Collect and compress instructions
        parts = self._collect_instruction_parts(thread)
        compressor = InstructionCompressor(self.context_budget)
        instructions = compressor.compress(**parts)

        # 4. Merge tools (user-provided + custom tools + skill tools)
        tools = list(self.tools)
        tools_used: list[str] = []
        if self.tool_loader:
            custom_tools = self.tool_loader.get_tools(tool_names)
            tools.extend(custom_tools)
            tools_used = [t.name for t in custom_tools]
        if self.skill_loader:
            tools.extend(self.skill_loader.get_tools())

        # 5. Register built-in tools
        tools.append(ReadRunDetailTool(thread.runs_dir))
        from runweave.runtime.workspace_tools import (
            ListFilesTool,
            ReadFileTool,
            WriteFileTool,
        )
        tools.append(WriteFileTool(thread.workspace_dir))
        tools.append(ReadFileTool(thread.workspace_dir))
        tools.append(ListFilesTool(thread.workspace_dir))

        # 6. Build CodeAgent
        context_callback = make_context_callback(self.context_budget)
        agent = CodeAgent(
            model=self.model,
            tools=tools,
            executor=executor,
            instructions=instructions,
            step_callbacks={ActionStep: context_callback},
            planning_interval=self.planning_interval,
            stream_outputs=self.stream_outputs,
        )

        return agent, thread, tools_used

    def _finalize_run(
        self,
        agent: CodeAgent,
        thread: Thread,
        task: str,
        smolagents_result: Any,
        tools_used: list[str],
    ) -> RunResult:
        """Post-process a completed run: persist memory, history, summary."""
        # Extract skills_used
        skills_used: list[str] = []
        if self.skill_loader:
            skills_used = self.skill_loader.load_skill_tool.get_loaded_and_reset()

        # Persist memory to disk
        save_memory(agent.memory, thread.memory_path)

        # Write run record and regenerate HISTORY.md
        history_writer = HistoryWriter(thread.runs_dir, thread.history_path)
        run_record = extract_run_record(
            run_number=history_writer.next_run_number(),
            task=task,
            memory=agent.memory,
            state=smolagents_result.state,
            skills_used=skills_used,
            output=smolagents_result.output,
            tools_used=tools_used,
        )
        history_writer.save_run(run_record)
        history_writer.generate_history()

        # Generate/update summary
        previous_summary = (
            thread.summary_path.read_text().strip()
            if thread.summary_path.is_file()
            else None
        )
        try:
            summary_gen = SummaryGenerator(self.model)
            summary = summary_gen.generate(
                task=task,
                output=smolagents_result.output,
                previous_summary=previous_summary,
            )
            thread.summary_path.write_text(summary)
        except Exception:
            logger.warning("Summary generation failed; using previous summary", exc_info=True)
            summary = previous_summary

        # Assemble RunResult
        return RunResult(
            output=smolagents_result.output,
            thread_id=thread.id,
            state=smolagents_result.state,
            step_count=_count_action_steps(smolagents_result.steps),
            token_usage=(
                smolagents_result.token_usage.dict()
                if smolagents_result.token_usage
                else None
            ),
            timing=(
                smolagents_result.timing.dict()
                if smolagents_result.timing
                else None
            ),
            summary=summary,
            skills_used=skills_used,
        )

    def _collect_instruction_parts(self, thread: Thread) -> dict:
        """Collect instruction components for InstructionCompressor."""
        skill_catalog = None
        if self.skill_loader:
            catalog = self.skill_loader.get_catalog()
            if catalog:
                skill_catalog = catalog

        # Load structured run records instead of reading HISTORY.md text
        history_writer = HistoryWriter(thread.runs_dir, thread.history_path)
        history_records = history_writer.load_records() or None

        thread_summary = None
        if thread.summary_path.is_file():
            summary = thread.summary_path.read_text().strip()
            if summary:
                thread_summary = summary

        return {
            "user_instructions": self.instructions,
            "skill_catalog": skill_catalog,
            "history_records": history_records,
            "thread_summary": thread_summary,
        }
