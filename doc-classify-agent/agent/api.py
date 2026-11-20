"""HTTP 服务：FastAPI + uvicorn。

约定：
    - 业务接口都要 X-API-Key，缺失/无效 401；
    - 结果按租户存 SQLite，跨租户取结果 403；同参复用返回 replayed=true；
    - 单请求超时 504；
    - 上传正文只在内存暂存，请求结束释放。

启动：
    python -m agent.api                         # 读 AGENT_HOST / AGENT_PORT
    uvicorn agent.api:app --host 0.0.0.0 --port 8081
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .agent import ClassifyAgent
from .auth import AuthError, authenticate
from .extract import ExtractError, extract
from .parse import list_docs, load_doc, register_upload, release_upload
from .store import TaskStore
from .validate import MAX_PAGES, ValidationError, check_text, decode_upload

SERVICE = "doc-classify-agent"
VERSION = "1.1"
TIMEOUT_S = float(os.environ.get("AGENT_TIMEOUT", "30"))
PROFILES = {"default", "legal", "report"}

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="task")
_STORE: TaskStore | None = None


def _store() -> TaskStore:
    """懒加载：import 时不建库文件，第一次真有请求才打开。"""
    global _STORE
    if _STORE is None:
        _STORE = TaskStore(os.environ.get("AGENT_DB") or None)
    return _STORE


def fingerprint_of(text: str, profile: str) -> str:
    """入参指纹：正文 + 分句配置。"""
    payload = json.dumps({"kind": "classify", "text": text, "profile": profile},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


app = FastAPI(
    title="doc-classify-agent",
    version=VERSION,
    description="文档敏感度分级：三路取证 + 加权融合 + 人工复核。",
)


# ---------------------------------------------------------------- 中间件与错误
@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex[:12]
    request.state.started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-Id"] = request.state.request_id
    elapsed = int((time.perf_counter() - request.state.started) * 1000)
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "request_id": request.state.request_id,
                      "route": f"{request.method} {request.url.path}",
                      "status": response.status_code, "elapsed_ms": elapsed},
                     ensure_ascii=False), flush=True)
    return response


def _error(request: Request, status: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}, "request_id": request_id},
        headers={"X-Request-Id": request_id},
    )


@app.exception_handler(ValidationError)
@app.exception_handler(ExtractError)
async def handle_domain_error(request: Request, exc):
    return _error(request, exc.status, exc.code, str(exc))


@app.exception_handler(RequestValidationError)
async def handle_body_error(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if any(e.get("type") == "json_invalid" for e in errors):
        return _error(request, 400, "bad_json", "请求体不是合法 JSON")
    fields = ", ".join(".".join(str(x) for x in e["loc"][1:]) for e in errors) or "请求体"
    return _error(request, 422, "bad_request", f"参数不合法：{fields}")


@app.exception_handler(AuthError)
async def handle_auth_error(request: Request, exc: AuthError):
    return _error(request, 401, exc.code, str(exc))


# ---------------------------------------------------------------- 鉴权
def require_tenant(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
    principal = authenticate({"x-api-key": x_api_key or ""})
    return principal.tenant


# ---------------------------------------------------------------- 路由
@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": SERVICE, "version": VERSION}


@app.get("/")
async def index():
    return {"service": SERVICE, "version": VERSION,
            "docs": "/docs", "classify": "POST /v1/classify"}


@app.post("/v1/classify")
async def classify(payload: dict, request: Request,
                   tenant: str = Depends(require_tenant)):
    doc_id, uploaded = _resolve_document(payload)
    profile = payload.get("profile") or "default"
    if profile not in PROFILES:
        raise ValidationError("profile 只支持 default / legal / report", 422, "bad_profile")

    try:
        doc_text = load_doc(doc_id, profile=profile).text
        fingerprint = fingerprint_of(doc_text, profile)

        hit = _store().find(tenant, "classify", fingerprint)
        if hit:
            body = dict(hit["body"])
            body["request_id"] = hit["id"]
            body["replayed"] = True
            return JSONResponse(content=body, headers={"X-Idempotent-Replay": "true"})

        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _EXECUTOR, lambda: ClassifyAgent(profile=profile).classify(doc_id)),
                timeout=TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise ValidationError(
                f"处理超过 {TIMEOUT_S:.0f}s，请缩小文档或重试", 504, "timeout") from None

        request_id = request.state.request_id
        body = result.to_dict()
        body["request_id"] = request_id
        body["replayed"] = False
        _store().save(request_id, tenant, "classify", fingerprint, body,
                      int((time.perf_counter() - request.state.started) * 1000))
        return body
    finally:
        if uploaded:
            release_upload(doc_id)


@app.get("/v1/results/{result_id}")
async def get_result(result_id: str, request: Request,
                     tenant: str = Depends(require_tenant)):
    entry = _store().get(result_id)
    if not entry:
        raise ValidationError("结果不存在或已过期", 404, "not_found")
    if entry["tenant"] != tenant:
        raise ValidationError("该结果属于其他租户", 403, "forbidden")
    body = dict(entry["body"])
    body["request_id"] = entry["id"]
    return body


# ---------------------------------------------------------------- 入参
def _resolve_document(payload: dict) -> tuple[str, bool]:
    if payload.get("file_b64"):
        filename, data = decode_upload(payload.get("filename", ""), payload["file_b64"])
        extracted = extract(filename, data)
        check_text(extracted.text, source="上传文件")
        if extracted.pages and extracted.pages > MAX_PAGES:
            raise ValidationError(
                f"文件 {extracted.pages} 页，超过上限 {MAX_PAGES} 页", 422, "too_many_pages")
        return register_upload(extracted.text, filename), True
    if payload.get("text") is not None:
        check_text(payload["text"])
        return register_upload(payload["text"]), True
    doc_id = payload.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValidationError("需要 doc_id、text 或 file_b64 三者之一", 422, "missing_document")
    if doc_id not in list_docs():
        raise ValidationError(
            f"文档不存在：{doc_id}（可用：{', '.join(list_docs())}）", 404, "doc_not_found")
    return doc_id, False


def main() -> int:
    import uvicorn

    host = os.environ.get("AGENT_HOST", "127.0.0.1")
    port = int(os.environ.get("AGENT_PORT", "8081"))
    if not os.environ.get("AGENT_API_KEYS"):
        print("[warn] 未设置 AGENT_API_KEYS，使用本地演示密钥 dev-key（仅限本机）", file=sys.stderr)
    print(f"{SERVICE} {VERSION} listening on http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
