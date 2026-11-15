"""日志脱敏：写日志时只留证据的指纹与长度，不留原文。

为什么必须有这个：
    两个项目处理的都是敏感文本（合同 / 敏感文档）。生产环境里日志通常会被采集、转发、
    长期留存，一旦原文进日志，等于在审计与备份系统里又存了一份敏感数据。
    调试需要原文时，靠 hash + 长度 + 条款号定位，再回业务库取原文并做权限校验。
"""

from __future__ import annotations

import hashlib
from typing import Any


def fingerprint(text: str, keep: int = 0) -> str:
    """返回 sha256 前 12 位；keep>0 时保留前 N 个字符（仅用于无敏感内容的场景）。"""
    if not text:
        return ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    if keep:
        return f"{text[:keep]}…#{digest}"
    return f"#{digest}"


def redact_text(text: str) -> dict[str, Any]:
    """把一段敏感文本替换成可检索但不含原文的描述。"""
    return {"sha256_12": fingerprint(text), "len": len(text)}


def redact_evidence(items: list[dict], keys: tuple[str, ...] = ("evidence", "sentence", "text")) -> list[dict]:
    """批量脱敏：命中上述 key 的字段替换为指纹对象。"""
    out: list[dict] = []
    for item in items:
        clean = dict(item)
        for k in keys:
            if isinstance(clean.get(k), str) and clean[k]:
                clean[k] = redact_text(clean[k])
        out.append(clean)
    return out
