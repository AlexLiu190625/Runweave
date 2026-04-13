from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from smolagents import LocalPythonExecutor


class WorkspaceExecutor(LocalPythonExecutor):
    """Executor that runs code inside a designated workspace directory.

    Not thread-safe: uses os.chdir() to switch the process-global cwd.
    This mirrors smolagents' own threading model — CodeAgent.run() is
    also not safe for concurrent use within a single process.
    """

    def __init__(
        self,
        workspace_dir: Path | str,
        additional_authorized_imports: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # Ensure the workspace directory exists
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            additional_authorized_imports=additional_authorized_imports or [],
            **kwargs,
        )

    def __call__(self, code_action: str) -> Any:
        # Switch to workspace directory before execution, restore afterwards
        prev_dir = Path.cwd()
        try:
            os.chdir(self.workspace_dir)
            return super().__call__(code_action)
        finally:
            os.chdir(prev_dir)
