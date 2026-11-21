"""把上传的文件转成纯文本：txt / md / docx / pdf。

放在校验之后、进业务之前。解析失败要和「文件类型不支持」分开报，
否则调用方分不清是自己传错了，还是文件本身没文本可提。
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from .validate import MAX_PAGES


class ExtractError(Exception):
    """解析失败：依赖缺失、文件损坏、加密、页数超限、没有可提取文本。"""

    def __init__(self, message: str, status: int = 422, code: str = "extract_failed"):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass
class Extracted:
    text: str
    pages: int | None = None      # pdf 能拿到真实页数，其余为 None


def _extract_txt(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ExtractError("文件不是 UTF-8 / GB18030 编码，请先转码", 422, "bad_encoding")


def _extract_docx(data: bytes) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ExtractError(
            "解析 docx 需要 python-docx（pip install -r requirements.txt）",
            501, "dependency_missing") from None
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 —— 损坏的包、非 docx 后缀伪装
        raise ExtractError(f"docx 解析失败：{exc}", 422, "corrupt_file") from None
    blocks: list[str] = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:                # 合同里的付款表、附件表都在表格里
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    return "\n".join(blocks)


def _extract_pdf(data: bytes) -> Extracted:
    try:
        try:
            import pymupdf as fitz          # PyMuPDF >= 1.24 的新导入名
        except ImportError:
            import fitz
    except ImportError:
        raise ExtractError(
            "解析 pdf 需要 PyMuPDF（pip install -r requirements.txt）",
            501, "dependency_missing") from None

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 —— 文件损坏或不是真 PDF
        raise ExtractError(f"pdf 解析失败：{exc}", 422, "corrupt_file") from None

    try:
        if document.needs_pass:
            raise ExtractError("文档已加密，无法解析；请先解密再上传", 422, "encrypted_file")
        pages = document.page_count
        if pages > MAX_PAGES:            # 先拦页数，避免把 1000 页全提一遍
            raise ExtractError(f"文件 {pages} 页，超过上限 {MAX_PAGES} 页", 422, "too_many_pages")
        parts: list[str] = []
        for index in range(pages):
            try:
                parts.append(document.load_page(index).get_text("text"))
            except Exception:  # noqa: BLE001 —— 单页坏掉不拖垮整份文档
                parts.append("")
        return Extracted(text="\n".join(parts).strip(), pages=pages)
    finally:
        document.close()


def extract(filename: str, data: bytes) -> Extracted:
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if suffix in {".txt", ".md"}:
        return Extracted(text=_extract_txt(data))
    if suffix == ".docx":
        return Extracted(text=_extract_docx(data))
    if suffix == ".pdf":
        result = _extract_pdf(data)
        if not result.text:
            raise ExtractError(
                "PDF 里没有可提取的文本，可能是扫描件；扫描件需要 OCR，当前不支持",
                422, "no_text_layer")
        return result
    raise ExtractError(f"暂不支持 {suffix or '无后缀'}", 415, "unsupported_type")
