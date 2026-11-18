"""编排：解析 → 三路并行 → 融合 → 输出（含证据与复核建议）。

这个类做三件事：
    - 三路互不依赖，可并行；这里用线程池模拟并发（生产是三个微服务并行调用）；
    - 每路都有独立的异常降级：某一路失败不影响另外两路；
    - 输出结构化结果：等级 + 各通道结论 + 命中证据 + 是否转人工。
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .channels import (ChannelResult, InvertedIndex, VectorIndex, build_indexes,
                       keyword_channel, level_rank, semantic_channel,
                       sentence_channel)
from .fusion import MIN_ACCEPT, DEFAULT_WEIGHTS, FusionResult, fuse, review_priority
from .memory import ReviewMemory, doc_fingerprint
from .parse import Doc, load_doc

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def load_json(name: str) -> dict | list:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@dataclass
class ClassifyResult:
    doc_id: str
    level: str
    confidence: float
    needs_review: bool
    priority: str
    channels: list[dict]
    evidence: list[dict]
    review_reasons: list[str]
    elapsed_ms: int
    errors: list[str] = field(default_factory=list)
    # --- L2 能力 ---
    questions: list[str] = field(default_factory=list)   # 要问复核人的具体问题
    retried: bool = False                                # 是否做过二次判定
    from_memory: bool = False                            # 是否沿用了历史人工结论

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "level": self.level,
            "confidence": round(self.confidence, 3),
            "needs_review": self.needs_review, "priority": self.priority,
            "channels": self.channels, "evidence": self.evidence,
            "review_reasons": self.review_reasons,
            "questions": self.questions, "retried": self.retried,
            "from_memory": self.from_memory,
            "elapsed_ms": self.elapsed_ms, "errors": self.errors,
        }


class ClassifyAgent:
    # 索引缓存：同一文档（同一分句配置）只构建一次。
    # 生产里索引进 ES / Milvus 是独立的服务，不会有这个问题；这里是复刻版的自救。
    _INDEX_CACHE: dict[tuple[str, str], tuple[Doc, InvertedIndex, VectorIndex]] = {}

    def __init__(self, *, weights: dict[str, float] | None = None,
                 channels: tuple[str, ...] = ("keyword", "sentence", "semantic"),
                 parallel: bool = True, rrf_k: int = 60,
                 profile: str = "default", min_accept: float = MIN_ACCEPT,
                 memory: ReviewMemory | None = None,
                 log_path: Path | None = None):
        self.lexicon = load_json("lexicons.json")
        self.rules = load_json("rules.json")
        self.weights = weights or DEFAULT_WEIGHTS
        self.enabled = channels
        self.parallel = parallel
        self.rrf_k = rrf_k
        self.profile = profile
        self.min_accept = min_accept
        self.memory = memory or ReviewMemory()
        self.log_path = Path(log_path) if isinstance(log_path, str) else (log_path or LOG_DIR / "runs.jsonl")

    def _prepare(self, doc_id: str) -> tuple[Doc, InvertedIndex, VectorIndex]:
        key = (doc_id, self.profile)
        cached = self._INDEX_CACHE.get(key)
        if cached is None:
            doc = load_doc(doc_id, profile=self.profile)
            inv, vec = build_indexes(doc)
            cached = (doc, inv, vec)
            self._INDEX_CACHE[key] = cached
        return cached

    # ------------------------------------------------------------ 单通道执行
    def _run_channel(self, name: str, doc: Doc,
                     inv: InvertedIndex, vec: VectorIndex,
                     threshold_scale: float = 1.0) -> ChannelResult:
        try:
            if name == "keyword":
                return keyword_channel(doc, self.lexicon)
            if name == "sentence":
                return sentence_channel(doc, self.lexicon)
            if name == "semantic":
                return semantic_channel(doc, self.rules, inv, vec,
                                        rrf_k=self.rrf_k, threshold_scale=threshold_scale)
            raise ValueError(f"未知通道：{name}")
        except Exception as exc:  # noqa: BLE001 —— 单路失败降级为空结果，不影响另外两路
            return ChannelResult(name, note=f"通道异常降级：{exc}")

    def _run_channels(self, doc: Doc, inv: InvertedIndex, vec: VectorIndex,
                      threshold_scale: float = 1.0) -> list[ChannelResult]:
        if self.parallel and len(self.enabled) > 1:
            with ThreadPoolExecutor(max_workers=len(self.enabled)) as pool:
                futures = [pool.submit(self._run_channel, n, doc, inv, vec, threshold_scale)
                           for n in self.enabled]
                return [f.result() for f in futures]
        return [self._run_channel(n, doc, inv, vec, threshold_scale) for n in self.enabled]

    # ------------------------------------------------------------ L2-a 追问清单
    @staticmethod
    def _questions(fused: FusionResult, channels: list[ChannelResult]) -> list[str]:
        qs: list[str] = []
        active = [c for c in channels if c.hits]
        if not active:
            qs.append("全文未命中任何词 / 句 / 规则，请确认是否按最低等级处理？")
            return qs
        disagree = sorted({c.level for c in active} - {fused.level}, key=level_rank)
        for lv in disagree:
            qs.append(f"有通道判为 {lv}、最终取 {fused.level}，请确认按哪个等级归档？")
        if [c.name for c in active] == ["semantic"]:
            rules = ", ".join(h.get("rule_id", "") for h in active[0].hits[:2])
            qs.append(f"仅语义规则命中（{rules}），缺少词 / 句级证据，请确认该条款是否成立？")
        if fused.level in {"L3", "L4"}:
            qs.append(f"结论为 {fused.level}，请确认定级依据是否充分？")
        return qs

    # ------------------------------------------------------------ 主流程
    def classify(self, doc_id: str, *, accept_level: str | None = None,
                 reviewer: str = "", note: str = "") -> ClassifyResult:
        started = time.perf_counter()
        doc, inv, vec = self._prepare(doc_id)

        channels = self._run_channels(doc, inv, vec)
        fused: FusionResult = fuse(channels, self.weights)
        retried = False

        # L2-b 二次判定：证据弱（仅语义命中 / 置信不足）时放宽阈值重跑一次语义通道
        weak = fused.confidence < self.min_accept or [c.name for c in channels if c.hits] == ["semantic"]
        if weak and "semantic" in self.enabled and not accept_level:
            relaxed = self._run_channels(doc, inv, vec, threshold_scale=0.85)
            fused2 = fuse(relaxed, self.weights)
            if fused2.confidence >= fused.confidence:
                channels, fused, retried = relaxed, fused2, True

        # L2-c 跨任务记忆：同一份文件人工判过，就沿用
        fingerprint = doc_fingerprint(doc.text)
        remembered = None if accept_level else self.memory.get(fingerprint)

        level, needs_review, reasons = fused.level, fused.needs_review, list(fused.reasons)
        from_memory = False
        if remembered:
            level, needs_review, from_memory = remembered["level"], False, True
            reasons.append(f"沿用历史人工结论：{remembered['level']}"
                           f"（{remembered.get('reviewer') or '未署名'}，{remembered['ts']}）")
        if accept_level:
            self.memory.remember(fingerprint, accept_level, reviewer=reviewer, note=note)
            level, needs_review = accept_level, False
            reasons.append(f"已记录人工复核结论：{accept_level}")

        questions = self._questions(fused, channels) if needs_review else []
        elapsed = int((time.perf_counter() - started) * 1000)

        result = ClassifyResult(
            doc_id=doc_id, level=level, confidence=fused.confidence,
            needs_review=needs_review, priority=review_priority(fused),
            channels=[c.to_dict() for c in channels],
            evidence=fused.evidence, review_reasons=reasons,
            elapsed_ms=elapsed, questions=questions, retried=retried,
            from_memory=from_memory,
            errors=[c.note for c in channels if "异常" in c.note],
        )
        self._log(result)
        return result

    # ------------------------------------------------------------ 日志
    def _log(self, result: ClassifyResult) -> None:
        """落盘前脱敏：证据只留指纹，不留文档原文。

        注意 deepcopy：to_dict 里嵌套的是同一批 dict，直接改会把内存中的返回值也改掉，
        调用方（CLI / 上层服务）就拿不到原文了。
        """
        import copy

        from .redact import redact_evidence

        record = copy.deepcopy(result.to_dict())
        record["evidence"] = redact_evidence(record.get("evidence", []))
        for ch in record.get("channels", []):
            ch["hits"] = redact_evidence(ch.get("hits", []))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rank(level: str) -> int:
    return level_rank(level)
