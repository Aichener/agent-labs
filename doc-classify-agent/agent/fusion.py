"""融合决策：三路结果按权重投票，冲突与低置信转人工。

设计原则（与生产一致）：
    1. 确定性证据（词 / 句，可追溯到库）权重高于语义证据；
    2. 只有语义命中、词句都没命中时，不直接采信——要么转人工，要么降级；
    3. 三路结论相差 ≥2 级视为冲突，强制人工复核；
    4. 高等级（L3/L4）一律要求人工确认，这是合规底线。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .channels import LEVELS, ChannelResult, level_rank

DEFAULT_WEIGHTS = {"keyword": 0.45, "sentence": 0.35, "semantic": 0.20}
MIN_ACCEPT = 0.45          # 最高分低于此值视为无法判定
HIGH_LEVELS = {"L3", "L4"}


@dataclass
class FusionResult:
    level: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    needs_review: bool = False
    reasons: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"level": self.level, "confidence": round(self.confidence, 3),
                "scores": {k: round(v, 3) for k, v in self.scores.items()},
                "needs_review": self.needs_review, "review_reasons": self.reasons,
                "evidence": self.evidence}


def fuse(channels: list[ChannelResult],
         weights: dict[str, float] | None = None,
         min_accept: float = MIN_ACCEPT) -> FusionResult:
    weights = weights or DEFAULT_WEIGHTS
    scores: dict[str, float] = {lv: 0.0 for lv in LEVELS}
    evidence: list[dict] = []
    active = [c for c in channels if c.hits]

    for ch in active:
        w = weights.get(ch.name, 0.0)
        if w <= 0:
            continue
        scores[ch.level] += w
        evidence.extend({**h, "channel": ch.name} for h in ch.hits[:5])

    total = sum(weights.get(c.name, 0.0) for c in active)
    total_all = sum(weights.values()) or 1.0

    if not active or total == 0:
        return FusionResult(level="L1", confidence=0.0, scores=scores,
                            needs_review=False,
                            reasons=["三路均未命中，判为最低等级（生产可配置抽样复核）"],
                            evidence=[])

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], -level_rank(kv[0])))
    best_level, best_score = ranked[0]
    # 置信度以「全部启用通道的权重和」为分母：三路一致 = 1.0，仅一路命中 = 该路权重
    confidence = best_score / total_all

    reasons: list[str] = []
    if best_score < min_accept:
        reasons.append(f"加权得分 {best_score:.2f} 低于阈值 {min_accept}（支持通道权重不足）")
    # 通道结论不一致：任何一路与最终结论不同，都必须人工确认（就不就高，是业务策略，不能自动降密）
    disagree = sorted({c.level for c in active} - {best_level}, key=level_rank)
    if disagree:
        reasons.append(f"通道结论不一致：最终 {best_level}，另有 {', '.join(disagree)}")
    # 只有语义命中：证据强度不足，转人工
    if [c.name for c in active] == ["semantic"]:
        reasons.append("仅语义规则命中，缺少词/句级确定性证据")
    if best_level in HIGH_LEVELS:
        reasons.append(f"{best_level} 属高等级，须人工确认")

    needs_review = bool(reasons)
    return FusionResult(level=best_level, confidence=confidence, scores=scores,
                        needs_review=needs_review, reasons=reasons, evidence=evidence)


def review_priority(result: FusionResult) -> str:
    """复核队列的排序：高等级优先，其次低置信。"""
    if result.level in {"L4", "L3"}:
        return "P0"
    if result.needs_review:
        return "P1"
    return "P2"