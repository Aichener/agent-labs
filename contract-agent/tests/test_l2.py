"""L2 能力测试：主动澄清、反思重规划、跨任务记忆、评测门禁。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.evaluate import GATE, check_gate, run_suite      # noqa: E402
from agent.llm import MockLLM                               # noqa: E402
from agent.loop import LoopAgent                            # noqa: E402
from agent.memory import MemoryStore                        # noqa: E402

LOG = ROOT / "logs" / "test_l2.jsonl"


def agent(**kw):
    return LoopAgent(MockLLM(), log_path=LOG, **kw)


class TestClarification(unittest.TestCase):
    """L2-a：信息不全时不硬猜，先提问。"""

    def test_missing_doc_asks(self):
        result = agent().run({"id": "q1", "doc_id": "", "rules": ["R1"]})
        self.assertEqual(result.status, "need_clarification")
        self.assertEqual(result.verdict, "pending")
        self.assertTrue(any("哪份合同" in q for q in result.questions))
        self.assertEqual(result.tool_calls, 0)

    def test_unknown_doc_asks(self):
        result = agent().run({"id": "q2", "doc_id": "doc_zzz", "rules": ["R1"]})
        self.assertEqual(result.status, "need_clarification")
        self.assertTrue(any("未找到合同" in q for q in result.questions))

    def test_unknown_rule_asks(self):
        result = agent().run({"id": "q3", "doc_id": "doc_a", "rules": ["R99"]})
        self.assertEqual(result.status, "need_clarification")
        self.assertTrue(any("R99" in q for q in result.questions))

    def test_answers_resume_the_run(self):
        result = agent().run({"id": "q4", "doc_id": "", "rules": []},
                             answers={"doc_id": "doc_a", "rules": ["R1"]})
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.verdict, "fail")   # doc_a 预付 50%
        self.assertIn("R1", result.failed_rules)


class TestReflection(unittest.TestCase):
    """L2-b：连续失败后带错误重规划，且只重规划一次。"""

    def test_replan_on_repeated_failures(self):
        import agent.tools as tools_mod

        original = tools_mod.TOOLS["check_rule"].fn
        state = {"calls": 0}

        def flaky(**kwargs):
            state["calls"] += 1
            if state["calls"] <= 2:          # 前两次运行时失败（参数合法，绕过计划校验）
                raise tools_mod.ToolError("模拟运行时失败：知识库连接超时")
            return original(**kwargs)

        tools_mod.TOOLS["check_rule"].fn = flaky
        try:
            result = LoopAgent(MockLLM(), log_path=LOG).run(
                {"id": "r1", "doc_id": "doc_a", "rules": ["R1", "R2"]})
        finally:
            tools_mod.TOOLS["check_rule"].fn = original

        self.assertEqual(result.replans, 1)
        self.assertGreaterEqual(state["calls"], 3)   # 失败两次 + 重规划后成功
        self.assertIn("R1", result.failed_rules)


class TestMemory(unittest.TestCase):
    """L2-c：人工改判写入记忆，下次规划带上。"""

    def setUp(self):
        self.mem_path = ROOT / "logs" / "memory_test.json"
        self.mem_path.unlink(missing_ok=True)
        self.mem = MemoryStore(self.mem_path)

    def test_correction_roundtrip(self):
        self.mem.remember_correction("t1", "doc_a", "R1", "pass", note="客户确认可接受")
        hints = self.mem.hints("t1")
        self.assertEqual(len(hints["corrections"]), 1)
        self.assertEqual(hints["corrections"][0]["human_verdict"], "pass")

    def test_run_uses_memory_and_writes_back(self):
        self.mem.remember_correction("t2", "doc_a", "R1", "pass")
        ag = LoopAgent(MockLLM(), tenant="t2", memory=self.mem, log_path=LOG)
        result = ag.run({"id": "m1", "doc_id": "doc_a", "rules": ["R1"]})
        self.assertTrue(result.memory_used)

        ag.run({"id": "m2", "doc_id": "doc_a", "rules": ["R1"], "human_verdict": "warn"})
        hints = self.mem.hints("t2")
        self.assertGreaterEqual(len(hints["corrections"]), 2)

    def test_memory_isolated_by_tenant(self):
        self.mem.remember_correction("t3", "doc_a", "R1", "fail")
        self.assertEqual(self.mem.hints("t4")["corrections"], [])


class TestGate(unittest.TestCase):
    """L2-d：门禁阈值判定（评测驱动）。"""

    def test_suite_meets_gate_on_mock(self):
        report = run_suite(MockLLM())
        self.assertEqual(check_gate(report["summary"]), [])

    def test_gate_rejects_low_success(self):
        bad = {"agent_success": 0.5, "baseline_success": 0.4,
               "plan_fallback_rate": 0.1, "avg_tool_calls": 4}
        failures = check_gate(bad)
        self.assertTrue(any("成功率" in f for f in failures))
        self.assertTrue(any("基线" in f for f in failures))

    def test_gate_rejects_high_fallback(self):
        bad = {"agent_success": 0.99, "baseline_success": 0.3,
               "plan_fallback_rate": 0.9, "avg_tool_calls": 4}
        self.assertTrue(any("兜底" in f for f in check_gate(bad)))

    def test_default_thresholds_sane(self):
        self.assertLess(GATE["min_success"], 1.0)
        self.assertGreater(GATE["min_gap_over_baseline"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
