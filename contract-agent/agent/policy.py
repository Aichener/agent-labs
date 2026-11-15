"""规则库与确定性判定。

判定不放在模型里。模型只负责规划调哪个工具，
规则命中、证据抽取、结论聚合全部走可测试的确定性代码。
这是这类合规场景能落地的前提：结论必须可复现、可审计。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    keyword: str              # 主关键词（检索/基线用）
    aliases: tuple[str, ...]  # 同义词，定位条款时任一命中即可
    required: bool            # True=缺条款即失败；False=约束型（未约定视为不适用）
    severity: str             # high / medium
    fix_template: str


RULES: dict[str, Rule] = {
    "R1": Rule("R1", "预付款比例不超过 30%", "预付",
               ("预付", "预付款", "首款", "首付款", "定金"), False, "high",
               "将预付款比例调整为不超过合同总额的 30%，剩余款项按验收节点分期支付。"),
    "R2": Rule("R2", "必须约定违约责任", "违约",
               ("违约",), True, "high",
               "增加违约责任条款，明确逾期交付与逾期付款的违约金比例"
               "（建议日万分之三，上限 20%）。"),
    "R3": Rule("R3", "争议解决方式必须明确", "争议",
               ("争议",), True, "high",
               "明确约定争议解决方式：提交××仲裁委员会仲裁，或向项目所在地人民法院起诉。"),
    "R4": Rule("R4", "保密期限须约定且不超过 5 年", "保密",
               ("保密",), True, "medium",
               "补充保密期限条款，建议约定为合同终止后 3 年，且不超过 5 年。"),
    "R5": Rule("R5", "必须约定验收标准与期限", "验收",
               ("验收",), True, "high",
               "补充验收标准与验收期限（如交付后 10 个工作日内完成验收，逾期视为通过）。"),
    "R6": Rule("R6", "必须约定知识产权归属", "知识产权",
               ("知识产权", "著作权", "专利权"), True, "medium",
               "补充知识产权条款，明确交付成果的著作权、专利权归属及后续使用许可范围。"),
}

ALL_RULE_IDS = list(RULES)

MISSING = "（全文未检出）"
NOT_APPLICABLE = "（未约定，视为不适用）"


@dataclass
class Finding:
    rule_id: str
    title: str
    status: str          # pass / fail
    severity: str
    evidence: str        # 命中的原文片段，或未检出的标记
    location: str        # 条款编号，如 第三条；缺失时为空
    suggestion: str      # 仅 fail 时给出

    def to_dict(self) -> dict:
        return asdict(self)


_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
_YEARS = re.compile(r"(\d{1,2})\s*年")


def _clauses_matching(clauses: list[dict], aliases: tuple[str, ...]) -> list[dict]:
    return [c for c in clauses if any(a in c["text"] for a in aliases)]


def _fail(rule: Rule, evidence: str, location: str) -> Finding:
    return Finding(rule.id, rule.title, "fail", rule.severity, evidence, location,
                   rule.fix_template)


def check_rule(rule_id: str, clauses: list[dict]) -> Finding:
    """对切分后的条款执行一条规则，返回可解释的结论。"""
    rule = RULES[rule_id]
    hits = _clauses_matching(clauses, rule.aliases)

    if not hits:
        # 约束型规则（如预付款比例）在「没有相关约定」时不算违规；其余视为缺条款
        if rule.required:
            return _fail(rule, MISSING, "")
        return Finding(rule.id, rule.title, "pass", rule.severity, NOT_APPLICABLE, "", "")

    first = hits[0]
    joined = "".join(c["text"] for c in hits)

    if rule_id == "R1":
        pcts = [float(x) for x in _PCT.findall(joined)]
        if not pcts or max(pcts) > 30:
            return _fail(rule, first["text"], first["no"])
        return Finding(rule.id, rule.title, "pass", rule.severity, first["text"], first["no"], "")

    if rule_id == "R2":
        return Finding(rule.id, rule.title, "pass", rule.severity, first["text"], first["no"], "")

    if rule_id == "R3":
        has_forum = re.search(r"仲裁|人民法院|法院", joined) is not None
        only_vague = re.search(r"协商解决|另行协商", joined) is not None
        if has_forum and not only_vague:
            return Finding(rule.id, rule.title, "pass", rule.severity,
                           first["text"], first["no"], "")
        return _fail(rule, first["text"], first["no"])

    if rule_id == "R4":
        years = [int(x) for x in _YEARS.findall(joined)]
        if not years or max(years) > 5:
            return _fail(rule, first["text"], first["no"])
        return Finding(rule.id, rule.title, "pass", rule.severity, first["text"], first["no"], "")

    # R5 / R6：关键词出现还不够，要看到「标准/期限」「归属/许可」这类限定词
    qualifier = {
        "R5": r"验收标准|验收期限|工作日|日内",
        "R6": r"归属|所有权|许可|归.{0,8}所有",
    }.get(rule_id, "")
    if qualifier and re.search(qualifier, joined) is None:
        return _fail(rule, first["text"], first["no"])
    return Finding(rule.id, rule.title, "pass", rule.severity, first["text"], first["no"], "")


def aggregate(findings: list[Finding]) -> dict:
    """把逐条结论合成最终判定。"""
    failed = [f for f in findings if f.status == "fail"]
    high = [f for f in failed if f.severity == "high"]
    if not failed:
        verdict, action = "pass", "可直接进入下一环节"
    elif high:
        verdict, action = "fail", "存在高风险条款，需修改后再签署"
    else:
        verdict, action = "warn", "存在中风险条款，建议协商修改"
    return {
        "verdict": verdict,
        "action": action,
        "failed_rules": [f.rule_id for f in failed],
        "findings": [f.to_dict() for f in findings],
    }


def fallback_plan(doc_id: str, rule_ids: list[str]) -> dict:
    """模型规划失败时的兜底计划：固定流程，保证任务不中断。"""
    steps = [{"tool": "extract_clauses", "args": {"doc_id": doc_id},
              "why": "兜底计划：先切分条款"}]
    steps += [{"tool": "check_rule", "args": {"doc_id": doc_id, "rule_id": r},
               "why": f"兜底计划：核查 {r}"} for r in rule_ids]
    steps.append({"tool": "finalize", "args": {}, "why": "兜底计划：汇总结论"})
    return {"goal": "核查合同风险条款（兜底计划）", "steps": steps, "fallback": True}
