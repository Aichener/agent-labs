"""规划器：让模型输出结构化计划，并做严格校验。

要点：
    - 计划必须是 JSON（schema 约束），不接受自由文本；
    - 校验失败 → 重试一次 → 仍失败 → 用兜底计划，任务不中断；
    - 规划器只决定「调什么、按什么顺序」，不做判断。
"""

from __future__ import annotations

import json
import re

from .llm import BaseLLM
from .policy import ALL_RULE_IDS, RULES, fallback_plan
from .tools import TOOLS, ToolError, list_docs, validate_args

PLANNER_SYSTEM = """你是合同核查 Agent 的规划器。
根据任务生成执行计划，**只输出 JSON**，格式：
{"goal": "...", "steps": [{"tool": "工具名", "args": {...}, "why": "一句话理由"}]}

可用工具：
- extract_clauses(doc_id)：切分条款（每个任务第一步必须调用）
- check_rule(doc_id, rule_id)：核查一条规则
- search_clauses(query, top_k?)：检索原文片段（仅在需要定位时调用）
- draft_fix(rule_id, evidence?)：给出修改建议
- finalize()：汇总结论（必须是最后一步）

约束：
1. 只使用上面列出的工具与参数，不要发明参数；
2. 先 extract_clauses，再逐条 check_rule，最后 finalize；
3. 不要输出解释文字、不要输出 Markdown 代码块。
"""


def build_planner_prompt(task: dict, hints: dict | None = None) -> str:
    rules = task.get("rules") or ALL_RULE_IDS
    rule_lines = "\n".join(f"- {r}：{RULES[r].title}" for r in rules if r in RULES)
    prompt = (
        f"doc_id={task['doc_id']}\n"
        f"任务：核查该合同的以下规则，并汇总结论\n{rule_lines}\n"
        f"（可用文档：{', '.join(list_docs())}）"
    )
    if hints:
        preferred = hints.get("preferred_rules") or []
        corrections = hints.get("corrections") or []
        notes = hints.get("notes") or []
        if preferred or corrections or notes:
            prompt += "\n\n【该租户的历史记忆（供参考，不得违反本次任务给的规则范围）】"
            if preferred:
                prompt += f"\n- 常用规则集：{', '.join(preferred)}"
            if notes:
                prompt += "\n- 备注：" + "；".join(notes[-3:])
            for c in corrections[-3:]:
                prompt += (f"\n- 人工改判：{c.get('doc_id')} 的 {c.get('rule_id')} "
                           f"曾由人工判为 {c.get('human_verdict')}")
    return prompt


def validate_plan(plan: dict, task: dict) -> list[str]:
    """返回问题列表，空列表表示通过。"""
    problems: list[str] = []
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return ["plan.steps 缺失或为空"]

    allowed_rules = set(task.get("rules") or ALL_RULE_IDS)
    tool_names = [s.get("tool") for s in steps if isinstance(s, dict)]

    if "extract_clauses" not in tool_names:
        problems.append("计划中缺少 extract_clauses")
    if tool_names and tool_names[-1] != "finalize":
        problems.append("计划最后一步必须是 finalize")
    if len(steps) > 15:
        problems.append(f"计划步骤过多：{len(steps)}")

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            problems.append(f"第 {i + 1} 步不是对象")
            continue
        name = step.get("tool")
        args = step.get("args") or {}
        if name not in TOOLS:
            problems.append(f"第 {i + 1} 步工具不存在：{name}")
            continue
        try:
            validate_args(name, args)
        except ToolError as exc:
            problems.append(f"第 {i + 1} 步参数非法：{exc}")
        if name == "check_rule":
            rid = args.get("rule_id")
            if rid and rid not in allowed_rules:
                problems.append(f"第 {i + 1} 步规则不在任务范围内：{rid}")
        if name == "extract_clauses" and args.get("doc_id") != task["doc_id"]:
            problems.append(f"第 {i + 1} 步 doc_id 与任务不一致")
    return problems


def make_plan(task: dict, llm: BaseLLM, hints: dict | None = None) -> tuple[dict, dict]:
    """返回 (plan, meta)。meta 记录是否兜底、模型原始输出的问题。"""
    meta = {"fallback": False, "problems": [], "raw": ""}
    try:
        raw = llm.complete(PLANNER_SYSTEM, build_planner_prompt(task, hints))
        meta["raw"] = raw[:2000]
        plan = _parse(raw)
    except Exception as exc:  # noqa: BLE001
        meta["fallback"] = True
        meta["problems"] = [f"模型规划失败：{exc}"]
        return fallback_plan(task["doc_id"], list(task.get("rules") or ALL_RULE_IDS)), meta

    problems = validate_plan(plan, task)
    if problems:
        # 一次修复机会：把问题告诉模型，让它重出
        try:
            raw2 = llm.complete(
                PLANNER_SYSTEM,
                build_planner_prompt(task, hints)
                + "\n\n上一次计划有以下问题，请修正后重新输出 JSON：\n"
                + "\n".join(f"- {p}" for p in problems),
            )
            plan2 = _parse(raw2)
            if not validate_plan(plan2, task):
                meta["problems"] = problems
                return plan2, meta
        except Exception:  # noqa: BLE001
            pass
        meta["fallback"] = True
        meta["problems"] = problems
        return fallback_plan(task["doc_id"], list(task.get("rules") or ALL_RULE_IDS)), meta

    return plan, meta


REFLECT_SYSTEM = """你是合同核查 Agent 的规划器，正在做失败后的重规划。
上一次计划的一部分工具调用失败了。请根据错误信息给出新的计划，**只输出 JSON**，格式：
{"goal": "...", "steps": [{"tool": "工具名", "args": {...}, "why": "一句话理由"}]}

要求：
1. 只使用：extract_clauses / check_rule / search_clauses / draft_fix / finalize；
2. 跳过已经失败的、不可能成功的步骤，换一种可行路径（比如改 doc_id、只重试失败的规则）；
3. 最后一步必须是 finalize；
4. 不要解释，不要 Markdown 代码块。
"""


def reflect_plan(task: dict, failures: list[str], llm: BaseLLM) -> dict | None:
    """L2-b 反思重规划：带着失败信息让模型重出计划；返回 None 表示重规划失败。"""
    try:
        raw = llm.complete(
            REFLECT_SYSTEM,
            f"doc_id={task['doc_id']}\n规则：{', '.join(task.get('rules') or ALL_RULE_IDS)}\n"
            f"失败记录：\n" + "\n".join(f"- {f}" for f in failures[-3:]),
        )
        plan = _parse(raw)
        if validate_plan(plan, task):
            return None
        return plan
    except Exception:  # noqa: BLE001
        return None


def _parse(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1:
        raise ValueError("输出中没有 JSON 对象")
    return json.loads(text[start:end + 1])
