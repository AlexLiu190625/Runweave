from __future__ import annotations


class NoCompatibleModelError(Exception):
    """Raised when Router cannot find any profile satisfying hard constraints."""

    def __init__(self, metadata, candidates_count: int) -> None:
        super().__init__(
            f"No model profile satisfies hard constraints for step "
            f"(kind={metadata.kind}, needs_long_context={metadata.needs_long_context}, "
            f"needs_structured_output={metadata.needs_structured_output}, "
            f"needs_tools={metadata.needs_tools}). "
            f"Candidates considered: {candidates_count}."
        )
        self.metadata = metadata
        self.candidates_count = candidates_count
