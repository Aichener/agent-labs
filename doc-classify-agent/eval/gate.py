"""质量门禁：融合必须明显优于最强单通道，且严重漏检为 0。

用法：
    python eval/gate.py          # 通过退出 0，不通过退出 1，并追加 eval/history.jsonl
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.evaluate import GATE, check_gate, run_all   # noqa: E402


def main() -> int:
    report = run_all()
    results = report["results"]
    failures = check_gate(results)

    fusion = results[-1]
    best_single = max(results[:3], key=lambda r: r["accuracy"])
    print(f"文档数 {len(report['docs'])}｜融合准确率 {fusion['accuracy']:.0%}"
          f"｜严重漏检 {fusion['severe_miss']:.0%}")
    print("-" * 56)
    print(f"最强单通道    {best_single['variant']} {best_single['accuracy']:.0%}")
    print(f"提升          {fusion['accuracy'] - best_single['accuracy']:.0%}"
          f"  (阈值 ≥{GATE['min_gap_over_best_single']:.0%})")
    print(f"融合准确率    {fusion['accuracy']:.0%}  (阈值 ≥{GATE['min_fusion_accuracy']:.0%})")
    print(f"严重漏检      {fusion['severe_miss']:.0%}  (阈值 ≤{GATE['max_severe_miss']:.0%})")
    print("-" * 56)

    history = ROOT / "eval" / "history.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "fusion": fusion["accuracy"],
                            "best_single": best_single["accuracy"],
                            "severe_miss": fusion["severe_miss"],
                            "failures": failures}, ensure_ascii=False) + "\n")

    if failures:
        print("门禁未通过：")
        for x in failures:
            print(f"  - {x}")
        return 1
    print("门禁通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
