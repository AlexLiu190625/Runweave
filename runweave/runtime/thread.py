from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Thread:
    """一个独立的工作单元，拥有自己的 workspace 和 memory 归档。"""

    # 唯一标识
    id: str
    # 创建时间（ISO 格式）
    created_at: str
    # workspace 目录，agent 在此读写文件
    workspace_dir: Path
    # memory 序列化文件路径
    memory_path: Path
    # summary 文件路径
    summary_path: Path
    # HISTORY.md 路径（结构化 run 历史）
    history_path: Path
    # runs/ 目录路径（每轮详细记录）
    runs_dir: Path
