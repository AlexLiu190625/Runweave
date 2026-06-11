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


class PlannerOutputError(Exception):
    """Raised when PlannerLLM cannot parse the model's output into a valid Plan."""

    def __init__(self, reason: str, raw_output: str | None = None) -> None:
        super().__init__(f"Planner output invalid: {reason}")
        self.reason = reason
        self.raw_output = raw_output


class UnsupportedPlanVersionError(Exception):
    """Raised when loading a plan.json whose version is not supported."""

    def __init__(self, found_version) -> None:
        super().__init__(
            f"Unsupported plan version: {found_version!r}. This Runweave "
            f"build only reads version=1 plans."
        )
        self.found_version = found_version
