"""单元测试：工具、规则、规划校验、循环护栏。

运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.llm import MockLLM                                   # noqa: E402
from agent.loop import LoopAgent                                # noqa: E402
from agent.planner import validate_plan                         # noqa: E402
from agent.policy import check_rule, aggregate                  # noqa: E402
from agent.tools import (ToolError, call_tool, list_docs,       # noqa: E402
                         load_doc, split_clauses, validate_args)


class TestClauses(unittest.TestCase):
    def test_split_has_seven_clauses(self):
        clauses = split_clauses(load_doc("doc_c"))
        self.assertEqual(len(clauses), 7)
        self.assertEqual(clauses[0]["no"], "第一条")
        self.assertIn("服务范围", clauses[0]["text"])

    def test_all_docs_parse(self):
        for doc_id in list_docs():
            self.assertGreaterEqual(len(split_clauses(load_doc(doc_id))), 5, doc_id)


class TestRules(unittest.TestCase):
    def _check(self, doc_id: str, rule_id: str):
        return check_rule(rule_id, split_clauses(load_doc(doc_id)))

    def test_r1_prepay_over_30_fails(self):
        self.assertEqual(self._check("doc_a", "R1").status, "fail")   # 50%
        self.assertEqual(self._check("doc_b", "R1").status, "pass")   # 20%
        self.assertEqual(self._check("doc_c", "R1").status, "pass")   # 30% 边界

    def test_r2_missing_penalty(self):
        self.assertEqual(self._check("doc_a", "R2").status, "fail")
        self.assertEqual(self._check("doc_b", "R2").status, "pass")

    def test_r3_vague_dispute(self):
        self.assertEqual(self._check("doc_d", "R3").status, "fail")   # 协商解决
        self.assertEqual(self._check("doc_c", "R3").status, "pass")   # 法院

    def test_r4_confidentiality_years(self):
        self.assertEqual(self._check("doc_c", "R4").status, "fail")   # 8 年
        self.assertEqual(self._check("doc_a", "R4").status, "pass")   # 3 年

    def test_r6_missing(self):
        self.assertEqual(self._check("doc_c", "R6").status, "fail")
        self.assertEqual(self._check("doc_d", "R6").status, "pass")

    def test_evidence_is_traceable(self):
        for doc_id in list_docs():
            text = load_doc(doc_id)
            for rule_id in ("R1", "R2", "R3", "R4", "R5", "R6"):
                f = self._check(doc_id, rule_id)
                if not f.evidence.startswith("（"):
                    self.assertIn(f.evidence, text, f"{doc_id}/{rule_id} 证据不可回溯")

    def test_aggregate_verdict(self):
        f_high = self._check("doc_a", "R1")
        self.assertEqual(aggregate([f_high])["verdict"], "fail")
        f_mid = self._check("doc_c", "R4")
        self.assertEqual(aggregate([f_mid])["verdict"], "warn")
        f_pass = self._check("doc_b", "R1")
        self.assertEqual(aggregate([f_pass])["verdict"], "pass")


class TestToolSafety(unittest.TestCase):
    def test_unknown_tool(self):
        with self.assertRaises(ToolError):
            call_tool("nope", {})

    def test_missing_required_arg(self):
        with self.assertRaises(ToolError):
            call_tool("check_rule", {"doc_id": "doc_a"})

    def test_extra_arg_rejected(self):
        with self.assertRaises(ToolError):
            validate_args("check_rule", {"doc_id": "doc_a", "rule_id": "R1", "foo": 1})

    def test_bad_doc(self):
        with self.assertRaises(ToolError):
            call_tool("extract_clauses", {"doc_id": "doc_zzz"})

    def test_unknown_rule(self):
        with self.assertRaises(ToolError):
            call_tool("check_rule", {"doc_id": "doc_a", "rule_id": "R99"})

    def test_search(self):
        out = call_tool("search_clauses", {"query": "保密 期限", "top_k": 2})
        self.assertTrue(out["ok"])
        self.assertLessEqual(len(out["data"]["hits"]), 2)


class TestPlanValidation(unittest.TestCase):
    def setUp(self):
        self.task = {"id": "t", "doc_id": "doc_a", "rules": ["R1", "R2"]}

    def test_valid_plan_passes(self):
        plan = {"steps": [
            {"tool": "extract_clauses", "args": {"doc_id": "doc_a"}},
            {"tool": "check_rule", "args": {"doc_id": "doc_a", "rule_id": "R1"}},
            {"tool": "finalize", "args": {}},
        ]}
        self.assertEqual(validate_plan(plan, self.task), [])

    def test_missing_extract(self):
        plan = {"steps": [{"tool": "finalize", "args": {}}]}
        self.assertTrue(validate_plan(plan, self.task))

    def test_last_step_not_finalize(self):
        plan = {"steps": [
            {"tool": "extract_clauses", "args": {"doc_id": "doc_a"}},
            {"tool": "check_rule", "args": {"doc_id": "doc_a", "rule_id": "R1"}},
        ]}
        self.assertTrue(validate_plan(plan, self.task))

    def test_out_of_scope_rule(self):
        plan = {"steps": [
            {"tool": "extract_clauses", "args": {"doc_id": "doc_a"}},
            {"tool": "check_rule", "args": {"doc_id": "doc_a", "rule_id": "R6"}},
            {"tool": "finalize", "args": {}},
        ]}
        self.assertTrue(validate_plan(plan, self.task))


class TestLoop(unittest.TestCase):
    def test_normal_run(self):
        task = {"id": "t1", "doc_id": "doc_a", "rules": ["R1", "R2", "R3"]}
        result = LoopAgent(MockLLM()).run(task)
        self.assertEqual(result.verdict, "fail")
        self.assertEqual(sorted(result.failed_rules), ["R1", "R2", "R3"])
        self.assertFalse(result.plan_fallback)
        self.assertEqual(result.tool_calls, 4)   # extract + 3 checks

    def test_tool_budget_guard(self):
        task = {"id": "t2", "doc_id": "doc_a",
                "rules": ["R1", "R2", "R3", "R4", "R5", "R6"]}
        result = LoopAgent(MockLLM(), max_tool_calls=2).run(task)
        self.assertIn("上限", result.stopped_by)
        self.assertLessEqual(result.tool_calls, 2)

    def test_duplicate_call_detected(self):
        class DupLLM(MockLLM):
            def _route(self, system, user):
                import json
                if "规划器" in system:
                    step = {"tool": "check_rule",
                            "args": {"doc_id": "doc_a", "rule_id": "R1"}}
                    return json.dumps({"steps": [
                        {"tool": "extract_clauses", "args": {"doc_id": "doc_a"}},
                        step, step, {"tool": "finalize", "args": {}}]})
                return "{}"

        result = LoopAgent(DupLLM()).run({"id": "t3", "doc_id": "doc_a", "rules": ["R1"]})
        self.assertIn("重复调用", result.stopped_by)

    def test_plan_fallback_on_garbage(self):
        class BadLLM(MockLLM):
            def complete(self, system, user, **kw):
                self.usage.add(1, 1)
                return "抱歉，我无法完成"

        task = {"id": "t4", "doc_id": "doc_b", "rules": ["R4", "R5"]}
        result = LoopAgent(BadLLM()).run(task)
        self.assertTrue(result.plan_fallback)
        self.assertEqual(result.verdict, "fail")   # 兜底计划仍能给出结论

    def test_no_findings_incomplete(self):
        task = {"id": "t5", "doc_id": "doc_a", "rules": ["R1"]}
        result = LoopAgent(MockLLM(), max_tool_calls=0).run(task)
        self.assertEqual(result.verdict, "incomplete")


if __name__ == "__main__":
    unittest.main(verbosity=2)
