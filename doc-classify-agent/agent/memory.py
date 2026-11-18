"""跨任务记忆：按文档内容指纹记住人工复核结论。

为什么按正文算指纹：
    同一份文档重新上传时 doc_id 可能变，但内容不变；用 sha256(正文) 做 key，
    才能真正做到「这份文件上次人工判过，这次直接沿用」。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEMORY_PATH = DATA_DIR / "review_memory.json"


def doc_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ReviewMemory:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else MEMORY_PATH

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def get(self, fingerprint: str) -> dict | None:
        return self._read().get(fingerprint)

    def remember(self, fingerprint: str, level: str, reviewer: str = "",
                 note: str = "", source: str = "human") -> None:
        data = self._read()
        data[fingerprint] = {
            "level": level, "reviewer": reviewer, "note": note, "source": source,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
