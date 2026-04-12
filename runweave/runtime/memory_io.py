from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents import AgentMemory


def save_memory(memory: AgentMemory, path: Path) -> None:
    """将 AgentMemory 序列化到 JSON 文件。

    使用 get_succinct_steps() 获取精简格式（排除 model_input_messages），
    序列化由 smolagents 内部的 .dict() 方法完成。
    保存的数据供用户查阅历史，不会在下次 run 时回灌到 agent。
    """
    steps = memory.get_succinct_steps()
    path.write_text(
        json.dumps(steps, ensure_ascii=False, indent=2, default=str)
    )


def load_memory(path: Path) -> list[dict]:
    """从 JSON 文件加载 memory 记录。

    返回原始 dict 列表，仅供查阅，不回灌到 agent 上下文。
    文件不存在时返回空列表。
    """
    if not path.is_file():
        return []
    return json.loads(path.read_text())
