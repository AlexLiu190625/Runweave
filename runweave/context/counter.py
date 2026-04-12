from __future__ import annotations

import math


class TokenCounter:
    """Character-ratio based token count estimator.

    Used for cross-run scenarios (estimating instruction sizes).
    For intra-run scenarios, use the actual token counts returned
    by smolagents.
    """

    # Conservative estimate: ~4 chars/token for English, ~2 for CJK,
    # using 3.5 as a compromise for mixed-language content
    CHARS_PER_TOKEN: float = 3.5

    @staticmethod
    def estimate(text: str | None) -> int:
        """Estimate the token count for a given text."""
        if not text:
            return 0
        return math.ceil(len(text) / TokenCounter.CHARS_PER_TOKEN)
