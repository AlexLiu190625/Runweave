from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from runweave.runtime.thread import Thread


class ThreadStore:
    """管理 thread 的磁盘布局。

    目录结构:
        <base_dir>/threads/<thread-id>/
            workspace/        ← agent 的工作目录
            memory.json       ← AgentMemory 序列化
            summary.txt       ← run 摘要（Stage 4）
            meta.json         ← {id, created_at}
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.threads_dir = base_dir / "threads"

    # ── 公开方法 ──────────────────────────────────────────

    def create(self, thread_id: str | None = None) -> Thread:
        """创建新 thread，生成目录结构并写入 meta.json。"""
        tid = thread_id or uuid.uuid4().hex[:12]
        created_at = datetime.now(timezone.utc).isoformat()

        thread_dir = self.threads_dir / tid
        thread_dir.mkdir(parents=True, exist_ok=True)
        (thread_dir / "workspace").mkdir(exist_ok=True)

        # 写入元信息
        meta = {"id": tid, "created_at": created_at}
        (thread_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )

        return self._build_thread(tid, created_at)

    def load(self, thread_id: str) -> Thread:
        """从磁盘加载已有 thread，不存在则抛出 FileNotFoundError。"""
        meta_path = self.threads_dir / thread_id / "meta.json"
        meta = json.loads(meta_path.read_text())
        return self._build_thread(meta["id"], meta["created_at"])

    def exists(self, thread_id: str) -> bool:
        """检查 thread 是否存在。"""
        return (self.threads_dir / thread_id / "meta.json").is_file()

    def list_threads(self) -> list[Thread]:
        """列出所有 thread，按创建时间倒序。"""
        if not self.threads_dir.is_dir():
            return []
        threads = []
        for meta_path in self.threads_dir.glob("*/meta.json"):
            meta = json.loads(meta_path.read_text())
            threads.append(self._build_thread(meta["id"], meta["created_at"]))
        threads.sort(key=lambda t: t.created_at, reverse=True)
        return threads

    # ── 内部方法 ──────────────────────────────────────────

    def _build_thread(self, thread_id: str, created_at: str) -> Thread:
        """根据 id 和创建时间构造 Thread 对象。"""
        thread_dir = self.threads_dir / thread_id
        return Thread(
            id=thread_id,
            created_at=created_at,
            workspace_dir=thread_dir / "workspace",
            memory_path=thread_dir / "memory.json",
            summary_path=thread_dir / "summary.txt",
        )
