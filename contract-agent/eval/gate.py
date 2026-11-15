"""质量门禁：跑评测，不达标就退出非零（用于 CI / 上线前检查）。

用法：
    python eval/gate.py                     # Mock 模型
    python eval/gate.py --llm openai        # 真模型
    python eval/gate.py --engine graph

通过标准见 agent/evaluate.py 的 GATE；每次结果追加到 eval/history.jsonl，便于看回归。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.evaluate import GATE, check_gate, run_suite   # noqa: E402
from agent.llm import build_llm                         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="mock", choices=["mock", "openai"])
    ap.add_argument("--engine", default="loop", choices=["loop", "graph"])
    args = ap.parse_args()

    llm = build_llm(args.llm)
    report = run_suite(llm, engine=args.engine)
    summary = report["summary"]
    failures = check_gate(summary)

    print(f"模型 {summary['model']}｜引擎 {summary['engine']}｜任务 {summary['tasks']}")
    print("-" * 56)
    print(f"成功率        {summary['agent_success']:.1%}  (阈值 ≥{GATE['min_success']:.0%})")
    print(f"基线          {summary['baseline_success']:.1%}")
    print(f"较基线提升    {summary['agent_success'] - summary['baseline_success']:.1%}"
          f"  (阈值 ≥{GATE['min_gap_over_baseline']:.0%})")
    print(f"计划兜底率    {summary['plan_fallback_rate']:.1%}"
          f"  (阈值 ≤{GATE['max_fallback_rate']:.0%})")
    print(f"平均工具调用  {summary['avg_tool_calls']}  (阈值 ≤{GATE['max_avg_tool_calls']})")
    print(f"P95 延迟      {summary['p95_latency_ms']} ms")
    print("-" * 56)

    history = ROOT / "eval" / "history.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "summary": summary, "failures": failures},
                           ensure_ascii=False) + "\n")

    if failures:
        print("门禁未通过：")
        for x in failures:
            print(f"  - {x}")
        return 1
    print("门禁通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
