"""文档解析：提取文本 → 分段 → 分句。

解析只做三件事：取文本、分段、分句。
    公开仓库不放 OCR 依赖，先支持 txt / md；docx / pdf 走可选依赖，
    解析结果统一成「段落 → 句子」两级结构——因为下游三路匹配都以句子为粒度。
"""

from __future__ import annotations

import time

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOC_DIR = DATA_DIR / "docs"

_WS = re.compile(r"[ \t\u3000]+")

DEFAULT_TERMINATORS = ("。", "？", "！", "；")


def load_parse_profiles() -> dict:
    path = DATA_DIR / "parse_rules.json"
    if not path.exists():
        return {"default": {"terminators": list(DEFAULT_TERMINATORS), "min_len": 1}}
    return json.loads(path.read_text(encoding="utf-8"))["profiles"]


@dataclass
class Doc:
    doc_id: str
    paragraphs: list[str]
    sentences: list[str]                    # 全局句子序列
    sent_index: list[tuple[int, int]] = field(default_factory=list)  # (段落, 该段内序号)

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


def extract_text(path: Path) -> str:
    """按扩展名提取纯文本。docx / pdf 为可选依赖，缺失时给出明确报错。

    这里读的是仓库里的语料文件；接口上传走的是 agent/extract.py，两边逻辑保持一致。
    """
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("解析 docx 需要 python-docx") from exc
        doc = Document(str(path))
        blocks = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    blocks.append(" | ".join(cells))
        return "\n".join(blocks)
    if suffix == ".pdf":
        try:
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("解析 pdf 需要 PyMuPDF") from exc
        document = fitz.open(stream=path.read_bytes(), filetype="pdf")
        try:
            if document.needs_pass:
                raise RuntimeError("pdf 已加密，无法解析")
            return "\n".join(document.load_page(i).get_text("text")
                             for i in range(document.page_count)).strip()
        finally:
            document.close()
    raise RuntimeError(f"不支持的格式：{suffix}")


_END_PUNCT = ("。", "！", "？", "；", "!", "?", ";")


def split_sentences(paragraph: str, terminators: tuple[str, ...] | None = None) -> list[str]:
    """按给定终止符切句，保留标点；只有「无标点的短残尾」才并入上一句。

    终止符可按文档画像配置（生产里由业务方按等级/文种确认），见 data/parse_rules.json。
    """
    terms = tuple(terminators or DEFAULT_TERMINATORS)
    pattern = "(?<=[" + "".join(re.escape(t) for t in terms) + "])"
    raw = [s.strip() for s in re.split(pattern, paragraph) if s.strip()]
    merged: list[str] = []
    for s in raw:
        dangling = not s.endswith(_END_PUNCT) and len(s) <= 4
        if merged and dangling:
            merged[-1] += s
        else:
            merged.append(s)
    return merged


def normalize(text: str) -> str:
    """归一化：去掉多余空白，用于精确匹配。"""
    return _WS.sub("", text).strip()


# 上传的正文只在内存里暂存，请求结束释放；不落盘，避免多租户互相覆盖
_TEXT_OVERRIDES: dict[str, str] = {}


def register_upload(text: str, name: str = "") -> str:
    doc_id = f"upload_{int(time.time() * 1000)}_{abs(hash(name + text)) % 100000}"
    _TEXT_OVERRIDES[doc_id] = text
    return doc_id


def release_upload(doc_id: str) -> None:
    _TEXT_OVERRIDES.pop(doc_id, None)


def load_doc(doc_id: str, profile: str = "default") -> Doc:
    if doc_id in _TEXT_OVERRIDES:
        text = _TEXT_OVERRIDES[doc_id]
    else:
        candidates = [p for p in DOC_DIR.glob(f"{doc_id}.*")
                      if p.suffix.lower() in (".txt", ".md", ".docx", ".pdf")]
        if not candidates:
            raise FileNotFoundError(f"文档不存在：{doc_id}（目录 {DOC_DIR}）")
        text = extract_text(candidates[0])
    profiles = load_parse_profiles()
    terms = tuple(profiles.get(profile, profiles["default"]).get("terminators", DEFAULT_TERMINATORS))
    paragraphs = [p.strip() for p in text.split("\n") if p.strip() and not p.strip().startswith("#")]
    sentences: list[str] = []
    sent_index: list[tuple[int, int]] = []
    for pi, para in enumerate(paragraphs):
        for si, sent in enumerate(split_sentences(para, terms)):
            sentences.append(sent)
            sent_index.append((pi, si))
    return Doc(doc_id=doc_id, paragraphs=paragraphs, sentences=sentences, sent_index=sent_index)


def list_docs() -> list[str]:
    found = {p.stem for p in DOC_DIR.iterdir() if p.suffix.lower() in (".txt", ".md", ".docx", ".pdf")}
    return sorted(found | set(_TEXT_OVERRIDES))
