"""L2 能力测试：追问清单、二次判定、跨任务记忆、评测门禁。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.agent import ClassifyAgent                        # noqa: E402
from agent.evaluate import check_gate, run_all              # noqa: E402
from agent.memory import ReviewMemory, doc_fingerprint      # noqa: E402
from agent.parse import load_doc                            # noqa: E402

LOG = ROOT / "logs" / "test_l2.jsonl"


def make_agent(memory: ReviewMemory | None = None) -> ClassifyAgent:
    return ClassifyAgent(memory=memory or ReviewMemory(ROOT / "logs" / "mem_l2.json"),
                         log_path=LOG)


class TestQuestions(unittest.TestCase):
    """L2-a：需要人确认时，给出具体问题而不是只给一个「需复核」。"""

    def test_conflict_generates_question(self):
        result = make_agent().classify("d2")     # 词库 L3 / 句与语义 L2
        self.assertTrue(result.needs_review)
        self.assertTrue(any("请确认按哪个等级" in q for q in result.questions))

    def test_no_hit_is_auto_passed(self):
        """三路全空按最低等级自动通过（生产可配抽样复核），因此不产生追问。"""
        result = make_agent().classify("d1")
        self.assertEqual(result.level, "L1")
        self.assertFalse(result.needs_review)
        self.assertEqual(result.questions, [])

    def test_high_level_generates_question(self):
        result = make_agent().classify("d3")     # L3 必复核
        self.assertTrue(any("定级依据" in q for q in result.questions))

    def test_no_review_no_question(self):
        memory = ReviewMemory(ROOT / "logs" / "mem_l2b.json")
        result = make_agent(memory).classify("d3", accept_level="L3", reviewer="张工")
        self.assertFalse(result.needs_review)
        self.assertEqual(result.questions, [])


class TestRetry(unittest.TestCase):
    """L2-b：证据弱时放宽阈值重跑一次，并标记 retried。"""

    def test_weak_evidence_triggers_retry(self):
        result = make_agent().classify("d7")     # 只有语义命中
        self.assertTrue(result.retried)

    def test_strong_evidence_no_retry(self):
        result = make_agent().classify("d3")     # 三路一致
        self.assertFalse(result.retried)


class TestMemory(unittest.TestCase):
    """L2-c：同一份文件内容不变，人工结论可复用。"""

    def setUp(self):
        self.mem = ReviewMemory(ROOT / "logs" / "mem_l2c.json")
        self.mem.clear()

    def test_accept_then_reuse(self):
        first = make_agent(self.mem).classify("d2", accept_level="L2", reviewer="张工")
        self.assertFalse(first.needs_review)
        self.assertEqual(first.level, "L2")

        second = make_agent(self.mem).classify("d2")
        self.assertTrue(second.from_memory)
        self.assertEqual(second.level, "L2")
        self.assertFalse(second.needs_review)

    def test_fingerprint_is_content_based(self):
        self.assertEqual(doc_fingerprint(load_doc("d3").text),
                         doc_fingerprint(load_doc("d3").text))
        self.assertNotEqual(doc_fingerprint(load_doc("d3").text),
                            doc_fingerprint(load_doc("d4").text))

    def test_memory_does_not_leak_across_docs(self):
        make_agent(self.mem).classify("d2", accept_level="L2")
        other = make_agent(self.mem).classify("d4")
        self.assertFalse(other.from_memory)


class TestGate(unittest.TestCase):
    """L2-d：门禁口径。"""

    def test_gate_passes_on_current_data(self):
        report = run_all()
        self.assertEqual(check_gate(report["results"]), [])

    def test_gate_rejects_weak_fusion(self):
        results = [
            {"variant": "仅敏感词", "accuracy": 0.75, "severe_miss": 0.12},
            {"variant": "仅敏感句", "accuracy": 0.62, "severe_miss": 0.38},
            {"variant": "仅语义规则", "accuracy": 0.75, "severe_miss": 0.12},
            {"variant": "三路融合（完整）", "accuracy": 0.80, "severe_miss": 0.0},
        ]
        failures = check_gate(results)
        self.assertTrue(any("提升" in f for f in failures))

    def test_gate_rejects_severe_miss(self):
        results = [
            {"variant": "仅敏感词", "accuracy": 0.9, "severe_miss": 0.0},
            {"variant": "仅敏感句", "accuracy": 0.8, "severe_miss": 0.0},
            {"variant": "仅语义规则", "accuracy": 0.7, "severe_miss": 0.0},
            {"variant": "三路融合（完整）", "accuracy": 0.99, "severe_miss": 0.12},
        ]
        self.assertTrue(any("漏检" in f for f in check_gate(results)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
