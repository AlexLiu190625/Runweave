from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from smolagents import LocalPythonExecutor


class WorkspaceExecutor(LocalPythonExecutor):
    """在指定 workspace 目录下执行代码的执行器。"""

    def __init__(
        self,
        workspace_dir: Path | str,
        additional_authorized_imports: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # 确保 workspace 目录存在
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            additional_authorized_imports=additional_authorized_imports or [],
            **kwargs,
        )

    def __call__(self, code_action: str) -> Any:
        # 切换到 workspace 目录后执行，结束后恢复原目录
        prev_dir = Path.cwd()
        try:
            os.chdir(self.workspace_dir)
            return super().__call__(code_action)
        finally:
            os.chdir(prev_dir)
