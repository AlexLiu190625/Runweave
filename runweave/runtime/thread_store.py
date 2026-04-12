from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from runweave.runtime.thread import Thread


class ThreadStore:
    """Manage thread disk layout.

    Directory structure:
        <base_dir>/threads/<thread-id>/
            workspace/        <- agent's working directory
            memory.json       <- serialized AgentMemory
            summary.txt       <- run summary
            meta.json         <- {id, created_at}
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.threads_dir = base_dir / "threads"

    # -- Public methods ------------------------------------------------

    def create(self, thread_id: str | None = None) -> Thread:
        """Create a new thread with directory structure and meta.json."""
        tid = thread_id or uuid.uuid4().hex[:12]
        created_at = datetime.now(timezone.utc).isoformat()

        thread_dir = self.threads_dir / tid
        thread_dir.mkdir(parents=True, exist_ok=True)
        (thread_dir / "workspace").mkdir(exist_ok=True)
        (thread_dir / "runs").mkdir(exist_ok=True)

        # Write metadata
        meta = {"id": tid, "created_at": created_at}
        (thread_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )

        return self._build_thread(tid, created_at)

    def load(self, thread_id: str) -> Thread:
        """Load an existing thread from disk. Raises FileNotFoundError if missing."""
        meta_path = self.threads_dir / thread_id / "meta.json"
        meta = json.loads(meta_path.read_text())
        return self._build_thread(meta["id"], meta["created_at"])

    def exists(self, thread_id: str) -> bool:
        """Check whether a thread exists."""
        return (self.threads_dir / thread_id / "meta.json").is_file()

    def list_threads(self) -> list[Thread]:
        """List all threads, sorted by creation time descending."""
        if not self.threads_dir.is_dir():
            return []
        threads = []
        for meta_path in self.threads_dir.glob("*/meta.json"):
            meta = json.loads(meta_path.read_text())
            threads.append(self._build_thread(meta["id"], meta["created_at"]))
        threads.sort(key=lambda t: t.created_at, reverse=True)
        return threads

    # -- Internal methods ----------------------------------------------

    def _build_thread(self, thread_id: str, created_at: str) -> Thread:
        """Construct a Thread object from id and creation time."""
        thread_dir = self.threads_dir / thread_id
        return Thread(
            id=thread_id,
            created_at=created_at,
            workspace_dir=thread_dir / "workspace",
            memory_path=thread_dir / "memory.json",
            summary_path=thread_dir / "summary.txt",
            history_path=thread_dir / "HISTORY.md",
            runs_dir=thread_dir / "runs",
        )
