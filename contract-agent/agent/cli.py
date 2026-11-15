"""命令行 demo：单份合同跑一遍，逐行打印计划、工具调用与结论。

用法：
    python -m agent.cli --doc doc_a                 # 离线 Mock，验证链路
    python -m agent.cli --doc doc_a --llm openai    # 接真模型（需 LLM_BASE_URL/LLM_MODEL）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm import LLMError, build_llm            # noqa: E402
from agent.loop import LoopAgent                     # noqa: E402
from agent.policy import ALL_RULE_IDS                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="合同/条款核查 Agent demo")
    ap.add_argument("--doc", required=True, help="文档 id，例如 doc_a")
    ap.add_argument("--llm", default="mock", choices=["mock", "openai"])
    ap.add_argument("--engine", default="loop", choices=["loop", "graph"],
                    help="loop=自研执行循环；graph=LangGraph 等价实现")
    ap.add_argument("--rules", default=",".join(ALL_RULE_IDS), help="逗号分隔的规则集")
    ap.add_argument("--tenant", default="default", help="租户名（跨任务记忆按租户隔离）")
    ap.add_argument("--answers", default="", help="补充信息（JSON），用于回答澄清问题")
    ap.add_argument("--human-verdict", default="", choices=["", "pass", "warn", "fail"],
                    help="人工结论，写入该租户记忆")
    ap.add_argument("--json", action="store_true", help="输出原始 JSON")
    args = ap.parse_args()

    task = {
        "id": f"cli-{args.doc}",
        "doc_id": args.doc,
        "rules": [r.strip() for r in args.rules.split(",") if r.strip()],
    }
    if args.human_verdict:
        task["human_verdict"] = args.human_verdict
    answers = json.loads(args.answers) if args.answers else None
    try:
        llm = build_llm(args.llm)
    except LLMError as exc:
        print(f"[LLM 初始化失败] {exc}", file=sys.stderr)
        return 2

    result = LoopAgent(llm, tenant=args.tenant).run(task, answers=answers)
    if args.engine == "graph":
        from agent.graph import run_graph

        out = run_graph(task, llm)
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0
        print("=" * 68)
        print(f"[LangGraph] 任务 {task['id']}｜结论 {out['verdict']}｜{out['action']}")
        print(f"未通过规则：{', '.join(out['failed_rules']) or '无'}"
              f"｜工具调用 {out['tool_calls']} 次｜{out['elapsed_ms']}ms")
        if out.get("stopped_by"):
            print(f"提前终止：{out['stopped_by']}")
        print("=" * 68)
        return 0
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if result.status == "need_clarification":
        print("=" * 68)
        print("需要补充信息后才能继续：")
        for q in result.questions:
            print(f"  ? {q}")
        print("=" * 68)
        return 0

    print("=" * 68)
    print(f"任务 {task['id']}｜文档 {task['doc_id']}｜规则 {', '.join(task['rules'])}")
    print(f"模型 {getattr(llm, 'name', '?')}｜计划兜底 {result.plan_fallback}"
          + (f"（{'; '.join(result.plan_problems)}）" if result.plan_problems else "")
          + (f"｜反思重规划 {result.replans} 次" if result.replans else "")
          + ("｜已用租户记忆" if result.memory_used else ""))
    print("-" * 68)
    for step in result.steps:
        flag = "OK " if step.ok else "ERR"
        note = f"  ← {step.note}" if step.note else ""
        print(f"[{flag}] {step.tool}({json.dumps(step.args, ensure_ascii=False)}) "
              f"{step.elapsed_ms}ms{note}")
        if step.error:
            print(f"      error: {step.error}")
    print("-" * 68)
    print(f"结论：{result.verdict}｜{result.action}")
    print(f"未通过规则：{', '.join(result.failed_rules) or '无'}")
    for f in result.findings:
        if f["status"] == "fail":
            print(f"  · {f['rule_id']} {f['title']}")
            print(f"    位置 {f['location'] or '—'}｜证据：{f['evidence'][:60]}")
            if f["suggestion"]:
                print(f"    建议：{f['suggestion']}")
    print("-" * 68)
    print(f"工具调用 {result.tool_calls} 次｜LLM {result.llm_calls} 次｜"
          f"tokens {result.tokens}｜{result.elapsed_ms}ms"
          + (f"｜提前终止：{result.stopped_by}" if result.stopped_by else ""))
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
