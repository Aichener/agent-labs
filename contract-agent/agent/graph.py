"""LangGraph 实现：同一套工具与护栏，换成图（StateGraph）来编排。

对照自研 loop，图的收益是「状态转移显式化」：
    plan → act → (条件边) → act … → finish
出问题时能直接看出卡在哪个节点、转移条件是否写错。

安装：pip install langgraph
用法：python -c "from agent.graph import run_graph; ..." 或 CLI --engine graph
"""

from __future__ import annotations

import json
import time
from typing import Any, TypedDict

from .llm import BaseLLM
from .planner import make_plan
from .policy import Finding, aggregate
from .tools import ToolError, call_tool

try:  # 允许在未安装 langgraph 的环境里正常导入本模块（只影响 graph 引擎）
    from langgraph.graph import END, StateGraph  # type: ignore

    HAS_LANGGRAPH = True
except Exception:  # noqa: BLE001
    HAS_LANGGRAPH = False


class GraphState(TypedDict, total=False):
    task: dict
    plan: dict
    idx: int
    findings: list[dict]
    errors: list[str]
    tool_calls: int
    stopped_by: str
    started: float
    verdict: str
    action: str
    failed_rules: list[str]
    plan_fallback: bool
    plan_problems: list[str]


def build_graph(llm: BaseLLM, *, max_tool_calls: int = 10, wall_timeout_s: float = 30.0):
    """构建并编译图。未安装 langgraph 时抛 RuntimeError。"""
    if not HAS_LANGGRAPH:
        raise RuntimeError("未安装 langgraph：pip install langgraph")

    def plan_node(state: GraphState) -> GraphState:
        task = state["task"]
        plan, meta = make_plan(task, llm)
        return {
            **state,
            "plan": plan,
            "idx": 0,
            "findings": [],
            "tool_calls": 0,
            "stopped_by": "",
            "started": time.perf_counter(),
            "plan_fallback": meta["fallback"],
            "plan_problems": meta["problems"],
        }

    def act_node(state: GraphState) -> GraphState:
        steps = state["plan"].get("steps", [])
        idx = state["idx"]
        if idx >= len(steps):
            return state
        step = steps[idx]
        tool, args = step.get("tool", ""), step.get("args") or {}

        findings = list(state["findings"])
        errors = list(state.get("errors", []))
        stopped_by = state["stopped_by"]
        tool_calls = state["tool_calls"]

        if time.perf_counter() - state["started"] > wall_timeout_s:
            stopped_by = f"超过墙钟超时 {wall_timeout_s}s"
        elif tool_calls >= max_tool_calls:
            stopped_by = f"达到工具调用上限 {max_tool_calls}"
        elif tool == "finalize":
            pass
        else:
            try:
                out = call_tool(tool, args)
                tool_calls += 1
                if tool == "check_rule":
                    d = out["data"]
                    findings.append(Finding(
                        rule_id=d["rule_id"], title=d["title"], status=d["status"],
                        severity=d["severity"], evidence=d["evidence"],
                        location=d["location"], suggestion=d["suggestion"],
                    ).to_dict())
            except ToolError as exc:
                # 单个工具失败不终止整个任务：记录后继续（与 loop 引擎行为一致）
                errors.append(f"{tool}: {exc}")

        return {**state, "idx": idx + 1, "findings": findings,
                "errors": errors, "tool_calls": tool_calls, "stopped_by": stopped_by}

    def should_continue(state: GraphState) -> str:
        if state.get("stopped_by"):
            return "finish"
        if state["idx"] >= len(state["plan"].get("steps", [])):
            return "finish"
        return "act"

    def finish_node(state: GraphState) -> GraphState:
        findings = [Finding(**f) for f in state["findings"]]
        if findings:
            summary = aggregate(findings)
            return {**state, "verdict": summary["verdict"],
                    "action": summary["action"],
                    "failed_rules": summary["failed_rules"]}
        return {**state, "verdict": "incomplete",
                "action": "未执行任何核查", "failed_rules": []}

    g = StateGraph(GraphState)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node("finish", finish_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "act")
    g.add_conditional_edges("act", should_continue, {"act": "act", "finish": "finish"})
    g.add_edge("finish", END)
    return g.compile()


def run_graph(task: dict, llm: BaseLLM, **kwargs: Any) -> dict:
    app = build_graph(llm, **kwargs)
    started = time.perf_counter()
    out = app.invoke({"task": task})
    return {
        "task_id": task["id"], "doc_id": task["doc_id"],
        "verdict": out.get("verdict"), "action": out.get("action"),
        "failed_rules": out.get("failed_rules", []),
        "findings": out.get("findings", []),
        "tool_calls": out.get("tool_calls", 0),
        "plan_fallback": out.get("plan_fallback", False),
        "stopped_by": out.get("stopped_by", ""),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


if __name__ == "__main__":
    from .llm import MockLLM

    print(json.dumps(run_graph({"id": "demo", "doc_id": "doc_a",
                                "rules": ["R1", "R2", "R3"]}, MockLLM()),
                     ensure_ascii=False, indent=2))
