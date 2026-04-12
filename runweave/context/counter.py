from __future__ import annotations

import math


class TokenCounter:
    """基于字符比例的 token 数估算器。

    用于 cross-run 场景（估算 instructions 大小）。
    intra-run 场景直接使用 smolagents 返回的实际 token 数。
    """

    # 保守估算：英文约 4 chars/token，中文约 2 chars/token，
    # 取 3.5 作为混合语言的折中值
    CHARS_PER_TOKEN: float = 3.5

    @staticmethod
    def estimate(text: str | None) -> int:
        """估算文本的 token 数。"""
        if not text:
            return 0
        return math.ceil(len(text) / TokenCounter.CHARS_PER_TOKEN)
