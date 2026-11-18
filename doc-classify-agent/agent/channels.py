"""三个匹配通道：敏感词、敏感句、语义规则。

三路匹配互不依赖：
    ① 敏感词 → ES 词库精准检索定级
    ② 敏感句 → ES 句库精准匹配定级
    ③ 语义规则 → 大模型 + 混合检索(Milvus/倒排) + RRF → 置信分定级

三路互不依赖，可以并行跑；任何一路单独失效都不应导致整体失效——
这正是要做融合而不是串行的原因。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .index import InvertedIndex, VectorIndex, rrf
from .parse import Doc, normalize

LEVELS = ["L1", "L2", "L3", "L4"]


def level_rank(level: str) -> int:
    return LEVELS.index(level) if level in LEVELS else 0


@dataclass
class ChannelResult:
    name: str
    level: str = "L1"          # 该通道给出的最高等级
    score: float = 0.0         # 通道自身的置信度（0-1）
    hits: list[dict] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "level": self.level,
                "score": round(self.score, 3), "hits": self.hits, "note": self.note}


# ------------------------------------------------------------------ ① 敏感词
def keyword_channel(doc: Doc, lexicon: dict) -> ChannelResult:
    hits: list[dict] = []
    for sid, sent in enumerate(doc.sentences):
        norm = normalize(sent)
        for item in lexicon["words"]:
            if item["word"] in norm:
                hits.append({"word": item["word"], "level": item["level"],
                             "sent_id": sid, "sentence": sent})
    if not hits:
        return ChannelResult("keyword", note="词库未命中")
    top = max(hits, key=lambda h: level_rank(h["level"]))
    return ChannelResult("keyword", level=top["level"],
                         score=min(1.0, 0.6 + 0.1 * len(hits)), hits=hits)


# ------------------------------------------------------------------ ② 敏感句
def sentence_channel(doc: Doc, lexicon: dict) -> ChannelResult:
    hits: list[dict] = []
    for sid, sent in enumerate(doc.sentences):
        norm = normalize(sent)
        for item in lexicon["sentences"]:
            target = normalize(item["text"])
            if target and (target in norm or norm in target):
                hits.append({"sentence": sent, "level": item["level"],
                             "sent_id": sid, "lib_text": item["text"]})
    if not hits:
        return ChannelResult("sentence", note="句库未命中")
    top = max(hits, key=lambda h: level_rank(h["level"]))
    return ChannelResult("sentence", level=top["level"], score=0.95, hits=hits)


# ------------------------------------------------------------------ ③ 语义规则
def semantic_channel(doc: Doc, rules: list[dict], inv: InvertedIndex,
                     vec: VectorIndex, rrf_k: int = 60,
                     threshold_scale: float = 1.0) -> ChannelResult:
    """每条规则跑一次：关键词召回 + 向量召回 → RRF → 取最佳句 → 置信分。

    threshold_scale < 1 表示放宽阈值（用于 L2 的「二次判定」：宁可多召回，交人工复核）。
    """
    hits: list[dict] = []
    for rule in rules:
        kw_rank = inv.search(" ".join(rule["keywords"]), top_k=5)
        vec_rank = vec.search(rule["desc"], top_k=5)
        fused = rrf([kw_rank, vec_rank], k=rrf_k)[:3]
        for sid, _rrf in fused:
            sent = doc.sentences[sid]
            coverage = sum(1 for kw in rule["keywords"] if kw in sent) / len(rule["keywords"])
            cosine = dict(vec.search(rule["desc"], top_k=len(doc.sentences) or 1)).get(sid, 0.0)
            score = 0.6 * coverage + 0.4 * cosine
            if score >= rule["threshold"] * threshold_scale:
                hits.append({"rule_id": rule["id"], "level": rule["level"],
                             "sent_id": sid, "sentence": sent,
                             "score": round(score, 3),
                             "coverage": round(coverage, 2),
                             "cosine": round(cosine, 3)})
                break  # 每条规则只取最佳句，避免同一规则刷分
    if not hits:
        return ChannelResult("semantic", note="语义规则未达阈值")
    top = max(hits, key=lambda h: (level_rank(h["level"]), h["score"]))
    return ChannelResult("semantic", level=top["level"],
                         score=top["score"], hits=hits,
                         note=f"命中 {len(hits)} 条规则")


def build_indexes(doc: Doc) -> tuple[InvertedIndex, VectorIndex]:
    return (InvertedIndex().build(doc.sentences), VectorIndex().build(doc.sentences))
