"""评测报告：Agent（多步工具）vs 基线（关键词规则引擎）。

口径定义在 agent/evaluate.py（与 eval/gate.py 共用，避免两套标准）。
用法：
    python eval/run_eval.py                    # Mock，快
    python eval/run_eval.py --llm openai       # 真模型
    python eval/run_eval.py --engine graph     # LangGraph 引擎
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.evaluate import run_suite            # noqa: E402
from agent.llm import build_llm                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="mock", choices=["mock", "openai"])
    ap.add_argument("--engine", default="loop", choices=["loop", "graph"])
    ap.add_argument("--out", default="eval/report.json")
    args = ap.parse_args()

    llm = build_llm(args.llm)
    report = run_suite(llm, engine=args.engine)
    summary, rows = report["summary"], report["rows"]

    print(f"模型：{summary['model']}｜引擎：{summary['engine']}｜任务数：{summary['tasks']}")
    print("-" * 62)
    print(f"Agent  成功率：{summary['agent_success']:.1%}")
    print(f"基线   成功率：{summary['baseline_success']:.1%}")
    print(f"平均工具调用：{summary['avg_tool_calls']} 次")
    print(f"平均耗时：{summary['avg_latency_ms']} ms｜P95：{summary['p95_latency_ms']} ms")
    print(f"计划兜底率：{summary['plan_fallback_rate']:.1%}｜tokens：{summary['tokens_total']}")
    print("-" * 62)

    bad = [r for r in rows if not r["agent_ok"]]
    if bad:
        print("未通过样例：")
        for r in bad[:8]:
            print(f"  {r['task']:<10} expect={r['expect']:<5} got={r['got']:<10} "
                  f"{'; '.join(r['reasons'])[:90]}")
    else:
        print("全部通过。")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {args.out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
