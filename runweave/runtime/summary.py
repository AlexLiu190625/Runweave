from __future__ import annotations

from typing import Any, TYPE_CHECKING

from smolagents.models import ChatMessage, MessageRole

if TYPE_CHECKING:
    from smolagents.models import Model

# 摘要生成的 system prompt
_SYSTEM_PROMPT = (
    "你是一个摘要助手。你的任务是维护一个 agent 工作线程的摘要，"
    "记录 agent 在多次运行中完成的工作。摘要应简洁（200-300 字以内），"
    "聚焦于：完成了什么任务、产生了什么文件或结果、当前工作状态。"
    "只输出摘要本身，不要输出任何额外说明。"
)


class SummaryGenerator:
    """用 LLM 为 thread 生成/更新摘要。"""

    def __init__(self, model: Model) -> None:
        self.model = model

    def generate(
        self,
        task: str,
        output: Any,
        previous_summary: str | None = None,
    ) -> str:
        """根据本次任务和输出，生成/更新 thread 摘要。

        有 previous_summary 时在其基础上追加更新；
        没有时从头生成。摘要会累积所有 run 的信息。
        """
        # 构造 user prompt
        parts: list[str] = []
        if previous_summary:
            parts.append(f"## 当前摘要\n{previous_summary}")
            parts.append("请在以上摘要的基础上，追加本次运行的信息。")
        else:
            parts.append("这是该线程的第一次运行，请生成初始摘要。")

        parts.append(f"## 本次任务\n{task}")
        parts.append(f"## 本次输出\n{output}")

        user_content = "\n\n".join(parts)

        # 调用 LLM
        response = self.model(messages=[
            ChatMessage(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_content),
        ])

        return response.content.strip()
