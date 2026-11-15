"""自研执行循环：规划 → 调用 → 观测 → 收敛。

没用 AgentExecutor，自己写循环：
    - 黑盒的 state / memory 出问题时定位不到是 prompt、工具还是循环逻辑；
    - 这里的每一步都要能打日志、能单测、能在护栏触发时体面退出。

护栏：
    1. 步数上限           —— 防止模型绕圈；
    2. 工具调用次数上限   —— 防止成本失控；
    3. 墙钟超时           —— 防止单任务卡死；
    4. 重复调用检测       —— 同样的工具 + 同样的参数第二次出现直接判定卡住；
    5. 工具失败重试一次   —— 参数里多余字段会被剥离后重试。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm import BaseLLM
from .memory import MemoryStore
from .planner import make_plan, reflect_plan
from .policy import ALL_RULE_IDS, RULES, Finding, aggregate
from .tools import ToolError, call_tool, list_docs

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


@dataclass
class StepTrace:
    index: int
    tool: str
    args: dict
    ok: bool
    elapsed_ms: int
    error: str = ""
    note: str = ""


@dataclass
class RunResult:
    task_id: str
    doc_id: str
    verdict: str = "incomplete"
    action: str = ""
    failed_rules: list[str] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    steps: list[StepTrace] = field(default_factory=list)
    tool_calls: int = 0
    llm_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    elapsed_ms: int = 0
    plan_ms: int = 0
    stopped_by: str = ""          # 触发护栏的原因
    plan_fallback: bool = False
    plan_problems: list[str] = field(default_factory=list)
    # --- L2 能力 ---
    status: str = "ok"            # ok / need_clarification
    questions: list[str] = field(default_factory=list)   # 需要用户补充的信息
    replans: int = 0              # 反思重规划次数
    memory_used: bool = False     # 本次是否用到了跨任务记忆

    def to_dict(self) -> dict:
        d = {
            "task_id": self.task_id, "doc_id": self.doc_id,
            "status": self.status, "questions": self.questions,
            "verdict": self.verdict, "action": self.action,
            "failed_rules": self.failed_rules, "findings": self.findings,
            "tool_calls": self.tool_calls, "llm_calls": self.llm_calls,
            "tokens": self.tokens, "cost_usd": round(self.cost_usd, 6),
            "elapsed_ms": self.elapsed_ms, "plan_ms": self.plan_ms, "stopped_by": self.stopped_by,
            "plan_fallback": self.plan_fallback, "plan_problems": self.plan_problems,
            "replans": self.replans, "memory_used": self.memory_used,
            "steps": [s.__dict__ for s in self.steps],
        }
        return d


class LoopAgent:
    def __init__(self, llm: BaseLLM, *, max_steps: int = 12, max_tool_calls: int = 10,
                 wall_timeout_s: float = 30.0, retry_tool_once: bool = True,
                 tenant: str = "default", memory: MemoryStore | None = None,
                 log_path: Path | None = None):
        self.llm = llm
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.wall_timeout_s = wall_timeout_s
        self.retry_tool_once = retry_tool_once
        self.tenant = tenant
        self.memory = memory or MemoryStore()
        self.log_path = log_path or (LOG_DIR / "runs.jsonl")

    # ------------------------------------------------------------ L2-a 主动澄清
    def _precheck(self, task: dict) -> list[str]:
        """信息不全就不要硬猜：直接生成要问用户的问题。"""
        questions: list[str] = []
        docs = list_docs()
        doc_id = task.get("doc_id")
        if not doc_id:
            questions.append(f"要核查哪份合同？可选：{', '.join(docs)}")
        elif doc_id not in docs:
            questions.append(f"未找到合同 {doc_id}，可选：{', '.join(docs)}")

        rules = task.get("rules")
        if not rules:
            questions.append(f"要核查哪些规则？可选：{', '.join(ALL_RULE_IDS)}")
        else:
            unknown = [r for r in rules if r not in RULES]
            if unknown:
                questions.append(f"规则 {unknown} 不存在，可选：{', '.join(ALL_RULE_IDS)}")
        return questions

    # ------------------------------------------------------------ 主流程
    def run(self, task: dict, answers: dict | None = None) -> RunResult:
        task = dict(task)
        if answers:
            task.update({k: v for k, v in answers.items() if v not in (None, "", [])})

        t0 = time.perf_counter()
        result = RunResult(task_id=task.get("id", "task"), doc_id=task.get("doc_id", ""))

        questions = self._precheck(task)
        if questions:
            result.status = "need_clarification"
            result.verdict = "pending"
            result.action = "信息不足，已向用户提问（未执行核查）"
            result.questions = questions
            self._log(result)
            return result

        hints = self.memory.hints(self.tenant)
        result.memory_used = bool(hints.get("corrections") or hints.get("preferred_rules"))

        plan, meta = make_plan(task, self.llm, hints=hints)
        result.plan_ms = int((time.perf_counter() - t0) * 1000)
        # 墙钟护栏只约束「执行阶段」：模型规划本身的耗时另算，避免冷启动把整单判死
        started = time.perf_counter()
        result.plan_fallback = meta["fallback"]
        result.plan_problems = meta["problems"]

        steps = list(plan.get("steps", []))
        findings: list[Finding] = []
        seen_calls: set[tuple[str, str]] = set()
        failures: list[str] = []

        i = 0
        while i < len(steps):
            if result.tool_calls >= self.max_tool_calls:
                result.stopped_by = f"达到工具调用上限 {self.max_tool_calls}"
                break
            if result.steps and len(result.steps) >= self.max_steps:
                result.stopped_by = f"达到步数上限 {self.max_steps}"
                break
            if time.perf_counter() - started > self.wall_timeout_s:
                result.stopped_by = f"超过墙钟超时 {self.wall_timeout_s}s"
                break

            step = steps[i]
            tool = step.get("tool", "")
            args = step.get("args") or {}

            signature = (tool, json.dumps(args, sort_keys=True, ensure_ascii=False))
            if signature in seen_calls:
                result.steps.append(StepTrace(i, tool, args, False, 0, "", "重复调用，已跳过"))
                result.stopped_by = result.stopped_by or f"检测到重复调用：{tool}"
                i += 1
                continue
            seen_calls.add(signature)

            if tool == "finalize":
                result.steps.append(StepTrace(i, "finalize", args, True, 0, "", "汇总结论"))
                i += 1
                continue

            trace, finding = self._execute(tool, args)
            result.tool_calls += 1
            result.steps.append(trace)
            if finding is not None:
                findings.append(finding)
            elif not trace.ok:
                failures.append(f"{tool}({json.dumps(args, ensure_ascii=False)}) → {trace.error}")

            # L2-b 反思重规划：连续两次失败 → 带着错误让模型重出计划（只做一次）
            if len(failures) >= 2 and result.replans == 0:
                new_plan = reflect_plan(task, failures, self.llm)
                if new_plan and new_plan.get("steps"):
                    steps = steps[: i + 1] + list(new_plan["steps"])
                    result.replans = 1
                    failures.clear()
                    seen_calls.clear()   # 重规划后允许重试之前的调用（否则会被重复检测拦住）
                else:
                    result.stopped_by = "重规划失败，保留已得结论"
                    break
            i += 1

        result.findings = [f.to_dict() for f in findings]
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        if findings:
            summary = aggregate(findings)
            result.verdict = summary["verdict"]
            result.action = summary["action"]
            result.failed_rules = summary["failed_rules"]
        else:
            result.verdict = "incomplete"
            result.action = "未执行任何核查，结果不可用"
            result.stopped_by = result.stopped_by or "计划中没有可执行的核查步骤"

        # L2-c 记忆回写：人工给了结论才记，避免把模型自己的判断写进记忆
        human = task.get("human_verdict")
        if human:
            for rid in task.get("rules") or []:
                self.memory.remember_correction(self.tenant, task["doc_id"], rid, human,
                                                note=task.get("human_note", ""))

        usage = getattr(self.llm, "usage", None)
        if usage is not None:
            result.tokens = usage.prompt_tokens + usage.completion_tokens
            result.cost_usd = usage.cost_usd
            result.llm_calls = usage.calls

        self._log(result)
        return result

    # ------------------------------------------------------------ 单步执行
    def _execute(self, tool: str, args: dict) -> tuple[StepTrace, Finding | None]:
        started = time.perf_counter()
        try:
            out = call_tool(tool, args)
        except ToolError as exc:
            if self.retry_tool_once:
                repaired = _repair_args(args)
                if repaired != args:
                    try:
                        out = call_tool(tool, repaired)
                        return (StepTrace(-1, tool, repaired, True,
                                          int((time.perf_counter() - started) * 1000),
                                          "", f"参数修复后重试成功（原错误：{exc}）"), None)
                    except ToolError as exc2:
                        return (StepTrace(-1, tool, repaired, False,
                                          int((time.perf_counter() - started) * 1000),
                                          str(exc2), "重试仍失败"), None)
            return (StepTrace(-1, tool, args, False,
                              int((time.perf_counter() - started) * 1000),
                              str(exc), "工具失败，已跳过"), None)

        elapsed = int((time.perf_counter() - started) * 1000)
        finding = None
        if tool == "check_rule":
            data = out["data"]
            finding = Finding(
                rule_id=data["rule_id"], title=data["title"], status=data["status"],
                severity=data["severity"], evidence=data["evidence"],
                location=data["location"], suggestion=data["suggestion"],
            )
        return StepTrace(-1, tool, args, True, elapsed, "", ""), finding

    # ------------------------------------------------------------ 日志
    def _log(self, result: RunResult) -> None:
        """落盘前脱敏：证据只留指纹，不留合同原文（生产日志会被采集与长期留存）。"""
        import copy

        from .redact import redact_evidence

        record = copy.deepcopy(result.to_dict())
        record["findings"] = redact_evidence(record.get("findings", []))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _repair_args(args: dict) -> dict:
    """极简参数修复：去掉常见多余字段，保证重试有意义。"""
    drop = {"reason", "why", "description", "note", "comment"}
    return {k: v for k, v in args.items() if k not in drop}


def run_task(task: dict, llm: BaseLLM, **kwargs: Any) -> RunResult:
    return LoopAgent(llm, **kwargs).run(task)
