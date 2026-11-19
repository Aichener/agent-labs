"""评测内核：通道消融 + 门禁阈值（run_eval 与 gate 共用同一口径）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .agent import ClassifyAgent
from .channels import level_rank
from .parse import list_docs

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    "仅敏感词": ("keyword",),
    "仅敏感句": ("sentence",),
    "仅语义规则": ("semantic",),
    "词+句（无语义）": ("keyword", "sentence"),
    "三路融合（完整）": ("keyword", "sentence", "semantic"),
}

# 门禁：融合必须明显优于最强单通道，且不允许出现严重漏检（低判 ≥2 级）
GATE = {
    "min_fusion_accuracy": 0.95,
    "max_severe_miss": 0.0,
    "min_gap_over_best_single": 0.10,
}


def load_truth() -> dict:
    return json.loads((ROOT / "data" / "ground_truth.json").read_text(encoding="utf-8"))


def run_variant(name: str, channels: tuple[str, ...], docs: list[str], truth: dict) -> dict:
    agent = ClassifyAgent(channels=channels, log_path=ROOT / "logs" / "eval.jsonl")
    rows, ok, severe, review = [], 0, 0, 0
    for doc_id in docs:
        result = agent.classify(doc_id)
        expect = truth[doc_id]["level"]
        gap = level_rank(expect) - level_rank(result.level)
        hit = result.level == expect
        ok += hit
        severe += gap >= 2
        review += result.needs_review
        rows.append({"doc": doc_id, "expect": expect, "got": result.level,
                     "ok": hit, "gap": gap, "review": result.needs_review,
                     "confidence": result.confidence, "retried": result.retried})
    n = len(docs)
    return {"variant": name, "accuracy": ok / n, "severe_miss": severe / n,
            "review_rate": review / n, "rows": rows}


def run_all() -> dict:
    docs = list_docs()
    truth = load_truth()
    results = [run_variant(name, ch, docs, truth) for name, ch in VARIANTS.items()]
    return {"docs": docs, "results": results}


def check_gate(results: list[dict], gate: dict | None = None) -> list[str]:
    g = gate or GATE
    fusion = results[-1]
    best_single = max(results[:3], key=lambda r: r["accuracy"])
    failures: list[str] = []
    if fusion["accuracy"] < g["min_fusion_accuracy"]:
        failures.append(f"融合准确率 {fusion['accuracy']:.0%} < {g['min_fusion_accuracy']:.0%}")
    if fusion["severe_miss"] > g["max_severe_miss"]:
        failures.append(f"融合严重漏检 {fusion['severe_miss']:.0%} > {g['max_severe_miss']:.0%}")
    gap = fusion["accuracy"] - best_single["accuracy"]
    if gap < g["min_gap_over_best_single"]:
        failures.append(
            f"较最强单通道（{best_single['variant']}）提升 {gap:.0%} < {g['min_gap_over_best_single']:.0%}")
    return failures
