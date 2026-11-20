"""API Key 鉴权：key → 租户。

生产环境应换成 OAuth2 / JWT + 密钥轮换 + 按接口授权，这里先把「谁能访问、属于哪个租户」立住。
密钥来自环境变量 AGENT_API_KEYS，格式 `key1:tenantA,key2:tenantB`。
没配置时给一个仅本地可用的 dev-key，方便演示；启动日志会提示。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEV_KEY = "dev-key"
DEV_TENANT = "demo"


class AuthError(Exception):
    def __init__(self, message: str = "缺少或无效的 API Key"):
        super().__init__(message)
        self.status = 401
        self.code = "unauthorized"


@dataclass(frozen=True)
class Principal:
    tenant: str
    key_id: str


def load_keys() -> dict[str, str]:
    raw = (os.environ.get("AGENT_API_KEYS") or "").strip()
    if not raw:
        return {DEV_KEY: DEV_TENANT}
    keys: dict[str, str] = {}
    for pair in raw.split(","):
        key, _, tenant = pair.partition(":")
        if key.strip() and tenant.strip():
            keys[key.strip()] = tenant.strip()
    return keys or {DEV_KEY: DEV_TENANT}


def authenticate(headers: dict[str, str], keys: dict[str, str] | None = None) -> Principal:
    """headers 用小写键：{'x-api-key': '...'}"""
    registry = keys if keys is not None else load_keys()
    key = (headers.get("x-api-key") or "").strip()
    if not key:
        raise AuthError("缺少 X-API-Key 请求头")
    tenant = registry.get(key)
    if not tenant:
        raise AuthError("API Key 无效")
    return Principal(tenant=tenant, key_id=key[:4] + "…")
