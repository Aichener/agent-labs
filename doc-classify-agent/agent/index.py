"""检索基础设施：倒排索引（替代 ES）+ 向量索引（替代 Milvus）+ RRF 融合。

为什么自己写而不是直接连 ES / Milvus：
    评测要可复现、要能在没有服务端的机器上跑。接口按生产抽象（index / search），
    换成 ES / Milvus 时只替换实现，三路通道与融合逻辑不动。
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

_TOKEN = re.compile(r"[\u4e00-\u9fa5]{1,4}|[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """粗粒度分词：中文按 2-4 字滑窗 + 英文数字串。够用于演示与评测。"""
    tokens: list[str] = []
    for m in _TOKEN.finditer(text):
        piece = m.group(0)
        if re.match(r"[\u4e00-\u9fa5]", piece):
            if len(piece) <= 2:
                tokens.append(piece)
            else:
                tokens.extend(piece[i:i + 2] for i in range(len(piece) - 1))
        else:
            tokens.append(piece.lower())
    return tokens


class InvertedIndex:
    """倒排索引：词 → 句子编号集合。够用，且没有外部依赖。"""

    def __init__(self) -> None:
        self._postings: dict[str, set[int]] = defaultdict(set)
        self._sentences: list[str] = []

    def build(self, sentences: list[str]) -> "InvertedIndex":
        self._sentences = list(sentences)
        for sid, sent in enumerate(self._sentences):
            for tok in set(tokenize(sent)):
                self._postings[tok].add(sid)
        return self

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        toks = tokenize(query)
        if not toks:
            return []
        scores: dict[int, float] = defaultdict(float)
        for tok in toks:
            for sid in self._postings.get(tok, ()):
                scores[sid] += 1.0 / math.log(2 + len(self._postings[tok]))
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        total = sum(s for _, s in ranked) or 1.0
        return [(sid, s / total) for sid, s in ranked]

    @property
    def size(self) -> int:
        return len(self._sentences)


class VectorIndex:
    """TF-IDF + cosine 的轻量向量索引，用来替稠密检索。"""

    def __init__(self) -> None:
        self._vectors: list[dict[str, float]] = []
        self._sentences: list[str] = []
        self._idf: dict[str, float] = {}

    def build(self, sentences: list[str]) -> "VectorIndex":
        self._sentences = list(sentences)
        df: dict[str, int] = defaultdict(int)
        for sent in sentences:
            for tok in set(tokenize(sent)):
                df[tok] += 1
        n = max(1, len(sentences))
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._vectors = [self._vectorize(s) for s in sentences]
        return self

    def _vectorize(self, text: str) -> dict[str, float]:
        tf: dict[str, int] = defaultdict(int)
        for tok in tokenize(text):
            tf[tok] += 1
        vec = {t: (1 + math.log(c)) * self._idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q = self._vectorize(query)
        sims: list[tuple[int, float]] = []
        for sid, vec in enumerate(self._vectors):
            sim = sum(w * vec.get(t, 0.0) for t, w in q.items())
            if sim > 0:
                sims.append((sid, sim))
        sims.sort(key=lambda kv: -kv[1])
        return sims[:top_k]


def rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：只按排名融合，规避不同检索器分值量纲不可比的问题。

    公式：score(d) = Σ 1 / (k + rank_i(d))，k 越大头部权重衰减越慢（常用 60）。
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
