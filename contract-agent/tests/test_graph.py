"""LangGraph 引擎与自研 loop 的一致性测试：同一任务必须给出同一结论。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.graph import HAS_LANGGRAPH, run_graph     # noqa: E402
from agent.llm import MockLLM                        # noqa: E402
from agent.loop import LoopAgent                     # noqa: E402


@unittest.skipUnless(HAS_LANGGRAPH, "未安装 langgraph，跳过图引擎测试")
class TestGraphParity(unittest.TestCase):
    TASKS = [
        {"id": "g1", "doc_id": "doc_a", "rules": ["R1", "R2", "R3"]},
        {"id": "g2", "doc_id": "doc_b", "rules": ["R4", "R5"]},
        {"id": "g3", "doc_id": "doc_c", "rules": ["R1", "R3", "R4", "R5"]},
        {"id": "g4", "doc_id": "doc_d", "rules": ["R1", "R2", "R3", "R4", "R5", "R6"]},
    ]

    def test_same_verdict_as_loop(self):
        for task in self.TASKS:
            with self.subTest(task=task["id"]):
                loop_result = LoopAgent(MockLLM()).run(task)
                graph_result = run_graph(task, MockLLM())
                self.assertEqual(loop_result.verdict, graph_result["verdict"])
                self.assertEqual(sorted(loop_result.failed_rules),
                                 sorted(graph_result["failed_rules"]))
                self.assertEqual(loop_result.tool_calls, graph_result["tool_calls"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
