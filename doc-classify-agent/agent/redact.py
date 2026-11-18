"""日志脱敏：写日志时只留证据指纹，不留原文（合同属于敏感文本）。"""

from __future__ import annotations

import hashlib
from typing import Any


def fingerprint(text: str) -> str:
    if not text:
        return ""
    return "#" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def redact_text(text: str) -> dict[str, Any]:
    return {"sha256_12": fingerprint(text), "len": len(text)}


def redact_evidence(items: list[dict], keys: tuple[str, ...] = ("evidence", "sentence", "text")) -> list[dict]:
    out: list[dict] = []
    for item in items:
        clean = dict(item)
        for k in keys:
            if isinstance(clean.get(k), str) and clean[k]:
                clean[k] = redact_text(clean[k])
        out.append(clean)
    return out
