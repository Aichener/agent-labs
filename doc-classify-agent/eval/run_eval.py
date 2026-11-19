"""通道消融报告：证明「为什么要融合」。

口径定义在 agent/evaluate.py（与 eval/gate.py 共用）。

用法：
    python eval/run_eval.py
    python eval/run_eval.py --out eval/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.evaluate import VARIANTS, run_all     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/report.json")
    args = ap.parse_args()

    report = run_all()
    docs = report["docs"]
    results = report["results"]
    truth = json.loads((ROOT / "data" / "ground_truth.json").read_text(encoding="utf-8"))

    print(f"文档数：{len(docs)}（{', '.join(docs)}）")
    print("-" * 72)
    print(f"{'方案':<20}{'等级准确率':>10}{'严重漏检率':>12}{'复核率':>10}{'二次判定':>10}")
    for r in results:
        retried = sum(1 for row in r["rows"] if row["retried"])
        print(f"{r['variant']:<20}{r['accuracy']:>10.0%}{r['severe_miss']:>12.0%}"
              f"{r['review_rate']:>10.0%}{retried:>10}")
    print("-" * 72)

    by_doc: dict[str, dict] = {}
    for r in results:
        for row in r["rows"]:
            by_doc.setdefault(row["doc"], {})[r["variant"]] = row["got"]
    print(f"{'文档':<6}{'真实':<6}" + "".join(f"{v:<16}" for v in VARIANTS))
    for doc_id in docs:
        line = f"{doc_id:<6}{truth[doc_id]['level']:<6}"
        for v in VARIANTS:
            got = by_doc[doc_id].get(v, "-")
            mark = "" if got == truth[doc_id]["level"] else "✗"
            line += f"{got + mark:<16}"
        print(line)
    print("-" * 72)

    full = results[-1]
    best_single = max(results[:3], key=lambda r: r["accuracy"])
    print(f"完整融合 {full['accuracy']:.0%} vs 最强单通道 {best_single['variant']} "
          f"{best_single['accuracy']:.0%}｜严重漏检 "
          f"{full['severe_miss']:.0%} vs {best_single['severe_miss']:.0%}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
