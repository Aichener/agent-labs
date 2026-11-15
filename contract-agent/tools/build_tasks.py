"""生成评测任务集（可复现）：4 份合同 × 5 组规则子集 = 20 条任务。

真正要扩到 50+ 条时，做法是加文档、加规则子集，而不是手写任务；
期望值来自 docs_meta.json 的 ground truth，避免「自己出题自己答」的偏差。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SUBSETS = [
    ["R1", "R2", "R3", "R4", "R5", "R6"],
    ["R1", "R4", "R6"],
    ["R2", "R3"],
    ["R5"],
    ["R1", "R3", "R4", "R5"],
]

HIGH = {"R1", "R2", "R3", "R5"}      # 与 policy.RULE severity 保持一致


def expected_verdict(failed: list[str]) -> str:
    if not failed:
        return "pass"
    return "fail" if any(r in HIGH for r in failed) else "warn"


def main() -> None:
    meta = json.loads((DATA / "docs_meta.json").read_text(encoding="utf-8"))
    tasks = []
    for doc_id, info in meta.items():
        for i, rules in enumerate(SUBSETS, start=1):
            failed = [r for r in rules if r in info["expect_fail"]]
            tasks.append({
                "id": f"{doc_id}-t{i}",
                "doc_id": doc_id,
                "rules": rules,
                "expect_fail": failed,
                "expect_verdict": expected_verdict(failed),
            })
    out = DATA / "tasks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} tasks -> {out}")


if __name__ == "__main__":
    main()
