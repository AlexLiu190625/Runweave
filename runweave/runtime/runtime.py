from __future__ import annotations

from pathlib import Path
from typing import Any

from smolagents import CodeAgent, Tool
from smolagents.memory import ActionStep
from smolagents.models import Model

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

# 默认数据目录
DEFAULT_BASE_DIR = Path.home() / ".runweave"


class Runtime:
    """Runweave 的顶层入口。

    将 smolagents CodeAgent 包装在持久化 thread 中：
    每次 run 都有独立的 workspace 和 memory 归档。
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
    ) -> None:
        self.model = model
        self.tools = tools or []
        self.instructions = instructions
        self.additional_authorized_imports = additional_authorized_imports or []
        self.store = ThreadStore(base_dir or DEFAULT_BASE_DIR)
        self.skill_loader = SkillLoader(skills_dir) if skills_dir else None
        self.tool_loader = ToolLoader(tools_dir) if tools_dir else None
        self.context_budget = context_budget or ContextBudget(model.model_id)

    def run(
        self,
        task: str,
        thread_id: str | None = None,
        tool_names: list[str] | None = None,
    ) -> RunResult:
        """在指定 thread 中执行任务，返回 RunResult。

        thread_id 为 None 时自动创建新 thread；
        传入已有 thread_id 可在同一 workspace 中继续工作。
        """
        # 1. 加载或创建 thread
        if thread_id and self.store.exists(thread_id):
            thread = self.store.load(thread_id)
        else:
            thread = self.store.create(thread_id)

        # 2. 构建 WorkspaceExecutor
        executor = WorkspaceExecutor(
            workspace_dir=thread.workspace_dir,
            additional_authorized_imports=self.additional_authorized_imports,
        )

        # 3. 收集并压缩 instructions
        parts = self._collect_instruction_parts(thread)
        compressor = InstructionCompressor(self.context_budget)
        instructions = compressor.compress(**parts)

        # 4. 合并 tools（用户传入的 + custom tools + skill tools）
        tools = list(self.tools)
        tools_used: list[str] = []
        if self.tool_loader:
            custom_tools = self.tool_loader.get_tools(tool_names)
            tools.extend(custom_tools)
            tools_used = [t.name for t in custom_tools]
        if self.skill_loader:
            tools.extend(self.skill_loader.get_tools())

        # 5. 注册 ReadRunDetailTool（按需加载历史 run 详情）
        tools.append(ReadRunDetailTool(thread.runs_dir))

        # 6. 构建 CodeAgent — smolagents 处理 agent loop 的一切
        context_callback = make_context_callback(self.context_budget)
        agent = CodeAgent(
            model=self.model,
            tools=tools,
            executor=executor,
            instructions=instructions,
            step_callbacks={ActionStep: context_callback},
        )

        # 7. 执行任务
        smolagents_result = agent.run(task, return_full_result=True)

        # 8. 提取 skills_used
        skills_used: list[str] = []
        if self.skill_loader:
            skills_used = self.skill_loader.load_skill_tool.get_loaded_and_reset()

        # 9. 持久化 memory 到磁盘
        save_memory(agent.memory, thread.memory_path)

        # 10. 写入 run 记录 + 重新生成 HISTORY.md
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

        # 11. 生成/更新 summary
        previous_summary = (
            thread.summary_path.read_text().strip()
            if thread.summary_path.is_file()
            else None
        )
        summary_gen = SummaryGenerator(self.model)
        summary = summary_gen.generate(
            task=task,
            output=smolagents_result.output,
            previous_summary=previous_summary,
        )
        thread.summary_path.write_text(summary)

        # 12. 组装 RunResult
        return RunResult(
            output=smolagents_result.output,
            thread_id=thread.id,
            state=smolagents_result.state,
            step_count=len(smolagents_result.steps),
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

    def _collect_instruction_parts(self, thread: Thread) -> dict[str, str | None]:
        """收集 instructions 的各组成部分，供 InstructionCompressor 压缩。"""
        skill_catalog = None
        if self.skill_loader:
            catalog = self.skill_loader.get_catalog()
            if catalog:
                skill_catalog = catalog

        history_md = None
        if thread.history_path.is_file():
            history = thread.history_path.read_text().strip()
            if history:
                history_md = history

        thread_summary = None
        if thread.summary_path.is_file():
            summary = thread.summary_path.read_text().strip()
            if summary:
                thread_summary = summary

        return {
            "user_instructions": self.instructions,
            "skill_catalog": skill_catalog,
            "history_md": history_md,
            "thread_summary": thread_summary,
        }
