"""命令行 demo：跑一份文档，打印三路通道结论、融合结果与证据。

用法：
    python -m agent.cli --doc d3
    python -m agent.cli --doc d6 --only keyword          # 只看单通道（消融）
    python -m agent.cli --doc d7 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import ClassifyAgent                    # noqa: E402
from agent.parse import list_docs                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="文档分级 Agent demo")
    ap.add_argument("--doc", required=True, help="文档 id，例如 d3")
    ap.add_argument("--only", default="", help="只跑指定通道：keyword / sentence / semantic")
    ap.add_argument("--profile", default="default",
                    help="分句配置：default / legal / report（见 data/parse_rules.json）")
    ap.add_argument("--accept", default="", choices=["", "L1", "L2", "L3", "L4"],
                    help="人工复核结论：写入该文档的记忆，下次直接沿用")
    ap.add_argument("--reviewer", default="", help="复核人（写入记忆）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    channels = tuple(args.only.split(",")) if args.only else ("keyword", "sentence", "semantic")
    agent = ClassifyAgent(channels=channels, profile=args.profile)
    result = agent.classify(args.doc, accept_level=args.accept or None,
                            reviewer=args.reviewer)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print("=" * 70)
    print(f"文档 {result.doc_id}｜通道 {', '.join(channels)}")
    print("-" * 70)
    for ch in result.channels:
        print(f"[{ch['name']:<8}] 结论 {ch['level']}  置信 {ch['score']:.2f}  {ch['note']}")
        for h in ch["hits"][:3]:
            text = h.get("sentence", "")[:44]
            print(f"           · {text}")
    print("-" * 70)
    print(f"最终等级：{result.level}｜置信 {result.confidence:.2f}｜"
          f"复核优先 {result.priority}｜{'需人工复核' if result.needs_review else '可自动通过'}")
    for r in result.review_reasons:
        print(f"  复核原因：{r}")
    if result.questions:
        print("需要确认：")
        for q in result.questions:
            print(f"  ? {q}")
    if result.retried:
        print("（做过二次判定：放宽语义阈值重跑一次）")
    if result.from_memory:
        print("（沿用历史人工结论）")
    if result.evidence:
        print("证据：")
        for e in result.evidence[:5]:
            src = e.get("word") or e.get("rule_id") or "句库"
            print(f"  · [{e.get('channel')}/{src}] {e.get('sentence', '')[:50]}")
    print("-" * 70)
    print(f"耗时 {result.elapsed_ms} ms｜可用文档：{', '.join(list_docs())}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
