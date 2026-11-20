"""任务持久化与幂等。

要解决的问题：
    - 服务重启后结果还在（之前放内存，重启就丢）；
    - 同样的入参重复提交，不重复跑模型、不重复花钱；
    - 换个租户提交同样的文档，互不影响（唯一约束带 tenant）。

用 SQLite 就够了：单文件、标准库自带、并发靠 WAL + 一把写锁。
生产换 Postgres 时，把这一层换掉即可，接口保持一致。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DB = DATA_DIR / "tasks.db"
RETENTION_DAYS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    tenant      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    status      TEXT NOT NULL,
    body        TEXT NOT NULL,
    elapsed_ms  INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_task_fp ON tasks(tenant, kind, fingerprint);
CREATE INDEX IF NOT EXISTS idx_task_created ON tasks(created_at);
"""


class TaskStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.commit()
        self._purge_expired()

    # ------------------------------------------------------------ 写
    def save(self, task_id: str, tenant: str, kind: str, fingerprint: str,
             body: dict, elapsed_ms: int, status: str = "done") -> None:
        """同租户同指纹已存在时保留旧记录（幂等），不覆盖。"""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO tasks"
                " (id, tenant, kind, fingerprint, status, body, elapsed_ms, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (task_id, tenant, kind, fingerprint, status,
                 json.dumps(body, ensure_ascii=False), elapsed_ms,
                 time.strftime("%Y-%m-%dT%H:%M:%S")),
            )
            self._conn.commit()

    # ------------------------------------------------------------ 读
    def find(self, tenant: str, kind: str, fingerprint: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE tenant=? AND kind=? AND fingerprint=?",
                (tenant, kind, fingerprint),
            ).fetchone()
        return self._row(row) if row else None

    def get(self, task_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return self._row(row) if row else None

    def count(self, tenant: str | None = None) -> int:
        with self._lock:
            if tenant:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM tasks WHERE tenant=?", (tenant,)).fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()
        return int(row["n"])

    def clear(self) -> None:
        """清空（测试与本地重置用）。"""
        with self._lock:
            self._conn.execute("DELETE FROM tasks")
            self._conn.commit()

    # ------------------------------------------------------------ 维护
    def _purge_expired(self) -> None:
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%S",
                               time.localtime(time.time() - RETENTION_DAYS * 86400))
        with self._lock:
            self._conn.execute("DELETE FROM tasks WHERE created_at < ?", (cutoff,))
            self._conn.commit()

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"], "tenant": row["tenant"], "kind": row["kind"],
            "status": row["status"], "elapsed_ms": row["elapsed_ms"],
            "created_at": row["created_at"],
            "body": json.loads(row["body"]),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
