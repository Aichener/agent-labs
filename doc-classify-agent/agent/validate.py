"""入参校验：文件类型、大小、页数、编码。

对外接口最容易出事的地方，所以校验放在最前面，错误码要能区分：
415 类型不支持 / 413 太大 / 422 内容不合规 / 400 请求格式错。
"""

from __future__ import annotations

import base64
import binascii
import math

MAX_BYTES = 5 * 1024 * 1024          # 单文件 5MB（演示用，生产可按客户调）
MAX_PAGES = 200                       # 按 1500 字/页估算
CHARS_PER_PAGE = 1500
ALLOWED_SUFFIXES = {".txt", ".md", ".docx", ".pdf"}
MAX_FIELD_LEN = 200


class ValidationError(Exception):
    def __init__(self, message: str, status: int, code: str):
        super().__init__(message)
        self.status = status
        self.code = code


def estimate_pages(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_PAGE))


def check_text(text: str, source: str = "text") -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValidationError(f"{source} 不能为空", 422, "empty_content")
    if "\x00" in text:
        raise ValidationError(f"{source} 含 NUL 字节，疑似二进制文件", 422, "binary_content")
    raw = text.encode("utf-8")
    if len(raw) > MAX_BYTES:
        raise ValidationError(
            f"{source} 为 {len(raw) // 1024}KB，超过上限 {MAX_BYTES // 1024}KB", 413, "too_large")
    pages = estimate_pages(text)
    if pages > MAX_PAGES:
        raise ValidationError(
            f"估算 {pages} 页，超过上限 {MAX_PAGES} 页", 422, "too_many_pages")


def decode_upload(filename: str, content_b64: str) -> tuple[str, bytes]:
    if not isinstance(filename, str) or not filename.strip():
        raise ValidationError("上传必须提供 filename", 400, "missing_filename")
    if len(filename) > MAX_FIELD_LEN:
        raise ValidationError("filename 过长", 400, "filename_too_long")
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise ValidationError(
            f"暂不支持 {suffix or '无后缀'}，当前支持：{', '.join(sorted(ALLOWED_SUFFIXES))}",
            415, "unsupported_type")
    try:
        data = base64.b64decode(content_b64, validate=True)
    except (binascii.Error, ValueError):
        raise ValidationError("file_b64 不是合法的 base64", 400, "bad_base64") from None
    if len(data) > MAX_BYTES:
        raise ValidationError(
            f"文件 {len(data) // 1024}KB，超过上限 {MAX_BYTES // 1024}KB", 413, "too_large")
    # 文本解码与页数判断交给 extract，这里只管「能不能收」
    return suffix, data
