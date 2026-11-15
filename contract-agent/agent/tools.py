"""工具层：每个工具都有明确的入参 schema、幂等性和错误类型。

工具是 Agent 的手脚。这里刻意做成「无副作用 + 可单测」：
    - 同样输入永远同样输出（幂等）；
    - 不写文件、不发网络请求，失败只抛 ToolError；
    - 每个工具都有 JSON Schema 供规划器约束参数，也用于运行时校验。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .policy import ALL_RULE_IDS, RULES, check_rule

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_DIR = DATA_DIR / "corpus"


class ToolError(RuntimeError):
    """工具执行失败（参数非法 / 数据不存在 / 内部错误）。可重试。"""


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[..., dict]


# ---------------------------------------------------------------- 数据访问

# 上传的合同暂存在内存里，请求结束就释放；不落盘，避免多租户互相覆盖
_UPLOADS: dict[str, str] = {}


def register_upload(text: str, name: str = "") -> str:
    """把 API 上传的正文注册成临时 doc_id，供工具层当普通文档读。"""
    doc_id = f"upload_{int(time.time() * 1000)}_{abs(hash(name + text)) % 100000}"
    _UPLOADS[doc_id] = text
    return doc_id


def release_upload(doc_id: str) -> None:
    _UPLOADS.pop(doc_id, None)


def load_doc(doc_id: str) -> str:
    if doc_id in _UPLOADS:
        return _UPLOADS[doc_id]
    path = CORPUS_DIR / f"{doc_id}.md"
    if not path.exists():
        raise ToolError(f"文档不存在：{doc_id}（可用：{', '.join(list_docs())}）")
    return path.read_text(encoding="utf-8")


def list_docs() -> list[str]:
    return sorted(p.stem for p in CORPUS_DIR.glob("*.md")) + sorted(_UPLOADS)


_CN = "一二三四五六七八九十"


def split_clauses(text: str) -> list[dict]:
    """按「第X条」切分；没有编号则按段落切分。"""
    pattern = re.compile(rf"第[{_CN}]{{1,3}}条")
    marks = list(pattern.finditer(text))
    clauses: list[dict] = []
    if marks:
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            body = text[m.start():end].strip()
            clauses.append({"no": m.group(0), "text": body})
    else:
        for i, para in enumerate(p for p in text.split("\n\n") if p.strip()):
            clauses.append({"no": f"段{i + 1}", "text": para.strip()})
    return clauses


# ---------------------------------------------------------------- 工具实现

def _extract_clauses(doc_id: str) -> dict:
    if not doc_id:
        raise ToolError("缺少参数 doc_id")
    clauses = split_clauses(load_doc(doc_id))
    if not clauses:
        raise ToolError(f"{doc_id} 未解析出任何条款")
    return {"doc_id": doc_id, "count": len(clauses), "clauses": clauses}


def _check_rule(doc_id: str, rule_id: str) -> dict:
    if rule_id not in RULES:
        raise ToolError(f"未知规则：{rule_id}（可用：{', '.join(ALL_RULE_IDS)}）")
    clauses = split_clauses(load_doc(doc_id))
    # 证据必须能在原文中按字面找到，否则视为抽取失败
    finding = check_rule(rule_id, clauses)
    # 证据必须能在原文中按字面找到，否则视为抽取失败（标记型 evidence 除外）
    if not finding.evidence.startswith("（") and finding.evidence not in load_doc(doc_id):
        raise ToolError(f"证据片段未在原文中命中：{rule_id}")
    return finding.to_dict()


def _search_clauses(query: str, top_k: int = 3) -> dict:
    if not query:
        raise ToolError("缺少参数 query")
    terms = [t for t in re.split(r"[\s,，。；;]+", query) if t]
    scored = []
    for doc_id in list_docs():
        for clause in split_clauses(load_doc(doc_id)):
            score = sum(1 for t in terms if t in clause["text"])
            if score:
                scored.append({"doc_id": doc_id, "no": clause["no"],
                               "text": clause["text"][:120], "score": score})
    scored.sort(key=lambda x: (-x["score"], x["doc_id"]))
    return {"query": query, "top_k": top_k, "hits": scored[:top_k]}


def _draft_fix(rule_id: str, evidence: str = "") -> dict:
    if rule_id not in RULES:
        raise ToolError(f"未知规则：{rule_id}")
    rule = RULES[rule_id]
    return {"rule_id": rule_id, "suggestion": rule.fix_template,
            "based_on": evidence[:80] or "（未检出相关条款）"}


TOOLS: dict[str, Tool] = {
    "extract_clauses": Tool(
        "extract_clauses", "把合同切分为条款，返回 doc_id 与条款列表",
        {"type": "object", "properties": {"doc_id": {"type": "string"}},
         "required": ["doc_id"]},
        lambda **kw: _extract_clauses(**kw),
    ),
    "check_rule": Tool(
        "check_rule", "对一份合同核查一条规则，返回 pass/fail、证据原文与条款位置",
        {"type": "object",
         "properties": {"doc_id": {"type": "string"}, "rule_id": {"type": "string"}},
         "required": ["doc_id", "rule_id"]},
        lambda **kw: _check_rule(**kw),
    ),
    "search_clauses": Tool(
        "search_clauses", "按关键词在全部合同中检索条款（用于定位相关原文）",
        {"type": "object",
         "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
         "required": ["query"]},
        lambda **kw: _search_clauses(**kw),
    ),
    "finalize": Tool(
        "finalize", "收口：汇总所有 finding 并给出最终判定（不产生新证据）",
        {"type": "object", "properties": {}},
        lambda **kw: {"finalized": True},
    ),
    "draft_fix": Tool(
        "draft_fix", "针对一条规则给出修改建议（模板 + 依据片段）",
        {"type": "object",
         "properties": {"rule_id": {"type": "string"}, "evidence": {"type": "string"}},
         "required": ["rule_id"]},
        lambda **kw: _draft_fix(**kw),
    ),
}


def validate_args(tool_name: str, args: dict) -> None:
    """按 schema 做最小校验：必填项、类型、未知参数。"""
    tool = TOOLS.get(tool_name)
    if tool is None:
        raise ToolError(f"未知工具：{tool_name}")
    schema = tool.schema
    for key in schema.get("required", []):
        if key not in args or args[key] in ("", None):
            raise ToolError(f"工具 {tool_name} 缺少必填参数 {key}")
    allowed = set(schema.get("properties", {}))
    extra = set(args) - allowed
    if extra:
        raise ToolError(f"工具 {tool_name} 收到未定义参数：{sorted(extra)}")
    for key, spec in schema.get("properties", {}).items():
        if key in args and spec.get("type") == "integer" and not isinstance(args[key], int):
            raise ToolError(f"参数 {key} 应为整数")


def call_tool(tool_name: str, args: dict) -> dict:
    """统一入口：校验 → 执行 → 计时。异常统一转成 ToolError。"""
    validate_args(tool_name, args)
    started = time.perf_counter()
    try:
        data = TOOLS[tool_name].fn(**args)
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"工具内部错误：{exc}") from exc
    return {"ok": True, "data": data, "elapsed_ms": int((time.perf_counter() - started) * 1000)}
