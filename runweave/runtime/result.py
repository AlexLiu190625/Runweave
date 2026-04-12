from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunResult:
    """Runtime.run() 的返回值。"""

    # agent 的最终输出
    output: Any
    # 所属 thread 的 ID
    thread_id: str
    # 执行状态："success" 或 "max_steps_error"
    state: str
    # 本次 run 执行的步数
    step_count: int
    # token 使用统计（可能为 None）
    token_usage: dict | None
    # 耗时统计（可能为 None）
    timing: dict | None
    # thread 摘要
    summary: str | None = None
    # 本次 run 加载的 skill 列表
    skills_used: list[str] = field(default_factory=list)
