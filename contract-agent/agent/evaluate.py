"""评测内核：任务集、基线、判定口径、批量运行。

被 eval/run_eval.py（人看的报告）和 eval/gate.py（CI 门禁）共用，
避免「报告一套口径、门禁另一套口径」。
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from .llm import BaseLLM
from .loop import LoopAgent
from .policy import RULES
from .tools import load_doc

ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = ROOT / "data" / "tasks.jsonl"
HIGH = {"R1", "R2", "R3", "R5"}

# 门禁阈值：低于任一项就不允许合并/上线
GATE = {
    "min_success": 0.95,       # 任务成功率
    "min_gap_over_baseline": 0.30,  # 至少比关键词基线高 30 个百分点
    "max_fallback_rate": 0.50,  # 计划兜底率（真模型下会高一些，放宽）
    "max_avg_tool_calls": 8.0,  # 平均工具调用
}


def load_tasks() -> list[dict]:
    return [json.loads(line) for line in TASKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def baseline(task: dict) -> dict:
    """基线：关键词规则引擎。只看关键词在不在，不切条款、不给证据、不给建议。"""
    text = load_doc(task["doc_id"])
    failed = [r for r in task["rules"] if RULES[r].keyword not in text]
    verdict = "pass" if not failed else ("fail" if any(r in HIGH for r in failed) else "warn")
    return {"verdict": verdict, "failed_rules": failed}


def judge(result, task: dict, doc_text: str) -> tuple[bool, list[str]]:
    """成功率口径：结论对 + 规则集合一致 + 证据可回溯 + fail 有建议。"""
    reasons: list[str] = []
    if result.verdict != task["expect_verdict"]:
        reasons.append(f"结论不一致：{result.verdict} != {task['expect_verdict']}")
    if sorted(result.failed_rules) != sorted(task["expect_fail"]):
        reasons.append(f"规则集合不一致：{result.failed_rules} != {task['expect_fail']}")
    for f in result.findings:
        if f["status"] != "fail":
            continue
        ev = f["evidence"]
        if not ev.startswith("（") and ev not in doc_text:
            reasons.append(f"{f['rule_id']} 证据无法回溯原文")
        if not f["suggestion"]:
            reasons.append(f"{f['rule_id']} 缺少修复建议")
    return (not reasons), reasons


def _runner(llm: BaseLLM, engine: str, **kw):
    if engine == "graph":
        from .graph import run_graph

        def run(task):
            out = run_graph(task, llm, **kw)
            return SimpleNamespace(
                verdict=out["verdict"], failed_rules=out["failed_rules"],
                findings=out["findings"], tool_calls=out["tool_calls"],
                tokens=0, plan_fallback=out.get("plan_fallback", False),
                stopped_by=out.get("stopped_by", ""), replans=0, status="ok",
            )

        return run

    agent = LoopAgent(llm, **kw)
    return agent.run


def run_suite(llm: BaseLLM, *, engine: str = "loop", log_path: Path | None = None) -> dict:
    tasks = load_tasks()
    run = _runner(llm, engine, log_path=log_path) if engine == "loop" else _runner(llm, engine)
    rows, ok, base_ok = [], 0, 0
    tool_calls, latencies, fallbacks, tokens = [], [], 0, 0

    for task in tasks:
        doc_text = load_doc(task["doc_id"])
        started = time.perf_counter()
        result = run(task)
        latency = int((time.perf_counter() - started) * 1000)

        passed, reasons = judge(result, task, doc_text)
        ok += passed
        b = baseline(task)
        b_ok = (b["verdict"] == task["expect_verdict"]
                and sorted(b["failed_rules"]) == sorted(task["expect_fail"]))
        base_ok += b_ok

        tool_calls.append(result.tool_calls)
        latencies.append(latency)
        fallbacks += bool(result.plan_fallback)
        tokens += getattr(result, "tokens", 0)
        rows.append({
            "task": task["id"], "doc": task["doc_id"],
            "expect": task["expect_verdict"], "got": result.verdict,
            "expect_fail": task["expect_fail"], "got_fail": result.failed_rules,
            "agent_ok": passed, "reasons": reasons,
            "baseline_ok": b_ok, "baseline_verdict": b["verdict"],
            "tool_calls": result.tool_calls, "ms": latency,
            "fallback": bool(result.plan_fallback),
            "replans": getattr(result, "replans", 0),
            "stopped_by": getattr(result, "stopped_by", ""),
        })

    n = len(tasks)
    hist = sorted(latencies)
    p95 = hist[max(0, int(len(hist) * 0.95) - 1)]
    summary = {
        "tasks": n,
        "agent_success": round(ok / n, 4),
        "baseline_success": round(base_ok / n, 4),
        "avg_tool_calls": round(statistics.mean(tool_calls), 2),
        "avg_latency_ms": round(statistics.mean(latencies), 1),
        "p95_latency_ms": p95,
        "plan_fallback_rate": round(fallbacks / n, 4),
        "tokens_total": tokens,
        "model": getattr(llm, "name", "?"),
        "engine": engine,
    }
    return {"summary": summary, "rows": rows}


def check_gate(summary: dict, gate: dict | None = None) -> list[str]:
    """返回未通过的项；空列表表示门禁通过。"""
    g = gate or GATE
    failures: list[str] = []
    gap = summary["agent_success"] - summary["baseline_success"]
    if summary["agent_success"] < g["min_success"]:
        failures.append(f"成功率 {summary['agent_success']:.1%} < {g['min_success']:.0%}")
    if gap < g["min_gap_over_baseline"]:
        failures.append(f"较基线提升 {gap:.1%} < {g['min_gap_over_baseline']:.0%}")
    if summary["plan_fallback_rate"] > g["max_fallback_rate"]:
        failures.append(f"计划兜底率 {summary['plan_fallback_rate']:.1%} > {g['max_fallback_rate']:.0%}")
    if summary["avg_tool_calls"] > g["max_avg_tool_calls"]:
        failures.append(f"平均工具调用 {summary['avg_tool_calls']} > {g['max_avg_tool_calls']}")
    return failures
