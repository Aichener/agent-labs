"""单元测试：解析、索引、三路通道、融合策略、端到端一致性。

运行：python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.agent import ClassifyAgent                          # noqa: E402
from agent.channels import (ChannelResult, build_indexes,      # noqa: E402
                            keyword_channel, level_rank,
                            semantic_channel, sentence_channel)
from agent.fusion import fuse                                  # noqa: E402
from agent.index import InvertedIndex, VectorIndex, rrf, tokenize  # noqa: E402
from agent.parse import list_docs, load_doc, split_sentences   # noqa: E402


class TestParse(unittest.TestCase):
    def test_split_sentences_keeps_punctuation(self):
        sents = split_sentences("第一句话。第二句话！第三句？")
        self.assertEqual(len(sents), 3)
        self.assertTrue(sents[0].endswith("。"))

    def test_short_fragment_merged(self):
        self.assertEqual(split_sentences("这里是完整的一句话。尾巴"), ["这里是完整的一句话。尾巴"])

    def test_every_doc_parses(self):
        for doc_id in list_docs():
            doc = load_doc(doc_id)
            self.assertTrue(doc.sentences, doc_id)


class TestIndex(unittest.TestCase):
    def setUp(self):
        self.sents = ["定型试验结论已通过评审。", "本周生产系统运行稳定。", "型号的联调进度延后。"]
        self.inv = InvertedIndex().build(self.sents)
        self.vec = VectorIndex().build(self.sents)

    def test_tokenize_chinese_bigrams(self):
        self.assertIn("定型", tokenize("定型试验"))

    def test_inverted_search_hits(self):
        hits = self.inv.search("定型试验")
        self.assertEqual(hits[0][0], 0)

    def test_vector_search_hits(self):
        hits = self.vec.search("试验结论评审")
        self.assertEqual(hits[0][0], 0)
        self.assertGreater(hits[0][1], 0)

    def test_rrf_prefers_consensus(self):
        # 两路都排第一的候选，融合后仍应第一
        r1 = [(2, 0.9), (0, 0.5)]
        r2 = [(2, 0.8), (1, 0.7)]
        self.assertEqual(rrf([r1, r2])[0][0], 2)

    def test_rrf_k_effect(self):
        r1 = [(0, 1.0), (1, 0.5)]
        small_k = dict(rrf([r1], k=1))
        large_k = dict(rrf([r1], k=100))
        # k 越小，头部优势越大
        self.assertGreater(small_k[0] - small_k[1], large_k[0] - large_k[1])


class TestChannels(unittest.TestCase):
    def setUp(self):
        self.doc = load_doc("d3")
        self.inv, self.vec = build_indexes(self.doc)
        self.lex = ClassifyAgent().lexicon
        self.rules = ClassifyAgent().rules

    def test_keyword_channel_finds_model(self):
        res = keyword_channel(self.doc, self.lex)
        self.assertEqual(res.level, "L3")
        self.assertTrue(res.hits)

    def test_sentence_channel_matches_library(self):
        res = sentence_channel(load_doc("d4"), self.lex)
        self.assertEqual(res.level, "L4")

    def test_semantic_channel_respects_threshold(self):
        res = semantic_channel(self.doc, self.rules, self.inv, self.vec)
        self.assertEqual(res.level, "L3")
        self.assertGreaterEqual(res.score, 0.5)

    def test_semantic_misses_unknown_phrasing(self):
        """d8 的表述不在规则覆盖范围内 —— 语义通道应漏掉。"""
        doc8 = load_doc("d8")
        inv, vec = build_indexes(doc8)
        res = semantic_channel(doc8, self.rules, inv, vec)
        self.assertEqual(res.hits, [])


class TestFusion(unittest.TestCase):
    def test_no_hits_returns_l1_without_review(self):
        r = fuse([ChannelResult("keyword"), ChannelResult("sentence"), ChannelResult("semantic")])
        self.assertEqual(r.level, "L1")
        self.assertFalse(r.needs_review)

    def test_disagreement_triggers_review(self):
        chans = [ChannelResult("keyword", "L3", 1.0, [{"sentence": "x"}]),
                 ChannelResult("sentence", "L2", 0.95, [{"sentence": "y"}])]
        r = fuse(chans)
        self.assertEqual(r.level, "L3")          # 词权重更高
        self.assertTrue(r.needs_review)
        self.assertTrue(any("不一致" in x for x in r.reasons))

    def test_high_level_requires_review(self):
        r = fuse([ChannelResult("keyword", "L4", 1.0, [{"sentence": "x"}])])
        self.assertTrue(r.needs_review)

    def test_semantic_only_requires_review(self):
        r = fuse([ChannelResult("semantic", "L4", 0.8, [{"sentence": "x"}])])
        self.assertTrue(r.needs_review)
        self.assertTrue(any("语义" in x for x in r.reasons))


class TestEndToEnd(unittest.TestCase):
    def test_ground_truth_accuracy(self):
        import json
        truth = json.loads((ROOT / "data" / "ground_truth.json").read_text(encoding="utf-8"))
        agent = ClassifyAgent(log_path=ROOT / "logs" / "test.jsonl")
        for doc_id, item in truth.items():
            with self.subTest(doc=doc_id):
                self.assertEqual(agent.classify(doc_id).level, item["level"])

    def test_parallel_equals_serial(self):
        p = ClassifyAgent(parallel=True, log_path=ROOT / "logs" / "t1.jsonl").classify("d6")
        s = ClassifyAgent(parallel=False, log_path=ROOT / "logs" / "t2.jsonl").classify("d6")
        self.assertEqual(p.level, s.level)
        self.assertEqual(p.confidence, s.confidence)

    def test_channel_degradation_isolated(self):
        """某一路异常时，其它路仍能给出结论（降级不扩散）。"""
        agent = ClassifyAgent(channels=("keyword", "bad_channel"),
                              log_path=ROOT / "logs" / "t3.jsonl")
        result = agent.classify("d3")
        self.assertEqual(result.level, "L3")
        self.assertTrue(any("异常" in e for e in result.errors))

    def test_single_channel_ablation_differs(self):
        only_kw = ClassifyAgent(channels=("keyword",), log_path=ROOT / "logs" / "t4.jsonl")
        self.assertEqual(only_kw.classify("d7").level, "L1")   # 词库抓不到 d7

    def test_level_rank_order(self):
        self.assertLess(level_rank("L1"), level_rank("L4"))


class TestLogRedaction(unittest.TestCase):
    def test_log_is_redacted_but_result_keeps_text(self):
        """回归测试：日志脱敏不能污染调用方拿到的结果（曾经因为浅拷贝踩过坑）。"""
        import json

        path = ROOT / "logs" / "redact_test.jsonl"
        path.unlink(missing_ok=True)
        result = ClassifyAgent(log_path=path).classify("d7")

        # 返回值里保留原文，供上层展示/复核
        self.assertIsInstance(result.evidence[0]["sentence"], str)
        self.assertTrue(result.evidence[0]["sentence"])

        # 落盘日志里只剩指纹
        record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertIsInstance(record["evidence"][0]["sentence"], dict)
        self.assertIn("sha256_12", record["evidence"][0]["sentence"])


class TestParseProfiles(unittest.TestCase):
    def test_legal_profile_does_not_split_on_question_mark(self):
        from agent.parse import split_sentences
        text = "甲方是否同意？乙方应在三日内答复。"
        default = split_sentences(text)
        legal = split_sentences(text, ("。", "；"))
        self.assertEqual(len(default), 2)
        self.assertEqual(len(legal), 1)

    def test_unknown_profile_falls_back(self):
        doc = load_doc("d3", profile="not-exist")
        self.assertTrue(doc.sentences)


if __name__ == "__main__":
    unittest.main(verbosity=2)
