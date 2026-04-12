from __future__ import annotations

from pathlib import Path
from typing import Any

from smolagents import CodeAgent, Tool
from smolagents.models import Model

from runweave.executor.workspace_executor import WorkspaceExecutor
from runweave.runtime.memory_io import save_memory
from runweave.runtime.result import RunResult
from runweave.runtime.summary import SummaryGenerator
from runweave.runtime.thread import Thread
from runweave.runtime.thread_store import ThreadStore
from runweave.skill.loader import SkillLoader

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
    ) -> None:
        self.model = model
        self.tools = tools or []
        self.instructions = instructions
        self.additional_authorized_imports = additional_authorized_imports or []
        self.store = ThreadStore(base_dir or DEFAULT_BASE_DIR)
        self.skill_loader = SkillLoader(skills_dir) if skills_dir else None

    def run(self, task: str, thread_id: str | None = None) -> RunResult:
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

        # 3. 拼接 instructions
        instructions = self._build_instructions(thread)

        # 4. 合并 tools（用户传入的 + skill tools）
        tools = list(self.tools)
        if self.skill_loader:
            tools.extend(self.skill_loader.get_tools())

        # 5. 构建 CodeAgent — smolagents 处理 agent loop 的一切
        agent = CodeAgent(
            model=self.model,
            tools=tools,
            executor=executor,
            instructions=instructions,
        )

        # 6. 执行任务
        smolagents_result = agent.run(task, return_full_result=True)

        # 7. 持久化 memory 到磁盘
        save_memory(agent.memory, thread.memory_path)

        # 8. 生成/更新 summary
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

        # 9. 组装 RunResult
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
        )

    def _build_instructions(self, thread: Thread) -> str | None:
        """拼接用户指令、skill 目录和 thread 摘要。"""
        parts: list[str] = []
        if self.instructions:
            parts.append(self.instructions)
        # 注入 skill 目录
        if self.skill_loader:
            catalog = self.skill_loader.get_catalog()
            if catalog:
                parts.append(catalog)
        # 注入之前 run 的摘要
        if thread.summary_path.is_file():
            summary = thread.summary_path.read_text().strip()
            if summary:
                parts.append(f"\n## 之前任务的摘要\n{summary}")
        return "\n\n".join(parts) if parts else None
