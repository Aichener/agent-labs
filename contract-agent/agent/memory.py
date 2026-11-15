"""跨任务记忆：按租户保存「规则偏好」与「人工改判」，规划时注入、运行后回写。

存什么：
    - 同一个客户/租户，第二次核查时不用再从零说明偏好规则；
    - 人改判过的条目（doc + 规则 + 结论）会作为提示给规划器，减少重复冲突；
    - 记忆只存结构化结论，不存合同原文（原文在业务库里）。

存储：data/memory.json，结构 {tenant: {"preferred_rules": [...], "corrections": [...]}}
"""

from __future__ import annotations

import json
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEMORY_PATH = DATA_DIR / "memory.json"
MAX_CORRECTIONS = 20


class MemoryStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else MEMORY_PATH

    # ------------------------------------------------------------ 读写
    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 —— 记忆损坏不应阻断核查
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, tenant: str) -> dict:
        return self._read().get(tenant, {})

    # ------------------------------------------------------------ 规划提示
    def hints(self, tenant: str) -> dict:
        data = self.get(tenant)
        return {
            "preferred_rules": data.get("preferred_rules", []),
            "corrections": data.get("corrections", [])[-5:],
            "notes": data.get("notes", []),
        }

    # ------------------------------------------------------------ 写入
    def remember_preference(self, tenant: str, key: str, value) -> None:
        all_data = self._read()
        entry = all_data.setdefault(tenant, {})
        entry.setdefault("preferred_rules", [])
        if key == "preferred_rules" and isinstance(value, list):
            entry["preferred_rules"] = value
        else:
            entry.setdefault("notes", []).append(f"{key}={value}")
        self._write(all_data)

    def remember_correction(self, tenant: str, doc_id: str, rule_id: str,
                            human_verdict: str, note: str = "") -> None:
        """记录人工改判：同一份合同同一条规则，人给的结论。"""
        all_data = self._read()
        entry = all_data.setdefault(tenant, {})
        corrections = entry.setdefault("corrections", [])
        corrections.append({
            "doc_id": doc_id, "rule_id": rule_id,
            "human_verdict": human_verdict, "note": note,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        entry["corrections"] = corrections[-MAX_CORRECTIONS:]
        self._write(all_data)

    def clear(self, tenant: str) -> None:
        all_data = self._read()
        all_data.pop(tenant, None)
        self._write(all_data)
