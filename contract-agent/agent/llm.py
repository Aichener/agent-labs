"""模型接入层：OpenAI 兼容接口 + 离线 Mock。零第三方依赖。

直接走 HTTP，不引 SDK：少一层依赖，评测环境不用联网装包；超时、重试、token 统计也都在自己手里，出错能定位到具体那次请求。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    """模型调用失败（网络、鉴权、返回格式）。"""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def add(self, prompt: int, completion: int, price_per_1k: float = 0.0) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1
        self.cost_usd += (prompt + completion) / 1000.0 * price_per_1k


class BaseLLM:
    name = "base"

    def complete(self, system: str, user: str, *, temperature: float = 0.0,
                 max_tokens: int = 1024) -> str:
        raise NotImplementedError

    # --- 便捷方法：要求模型输出 JSON，失败重试一次并剥离代码块 ---
    def json_complete(self, system: str, user: str, *, retries: int = 1) -> dict:
        last: Exception | None = None
        for attempt in range(retries + 1):
            raw = self.complete(system, user)
            try:
                return _parse_json(raw)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < retries:
                    user = user + "\n\n上一次输出不是合法 JSON，请只输出 JSON，不要解释。"
        raise LLMError(f"模型未返回合法 JSON：{last}")


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no json object found")
    return json.loads(text[start:end + 1])


class OpenAICompatLLM(BaseLLM):
    """任何 OpenAI 兼容端点：Ollama(/v1)、DeepSeek、vLLM、DashScope 兼容模式等。"""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or "http://127.0.0.1:11434/v1").rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY") or "not-needed"
        self.model = model or os.getenv("LLM_MODEL") or "qwen2.5:7b"
        self.timeout = timeout
        self.name = f"openai-compat:{self.model}"
        self.usage = Usage()
        self.price_per_1k = float(os.getenv("LLM_PRICE_PER_1K", "0"))

    def complete(self, system: str, user: str, *, temperature: float = 0.0,
                 max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise LLMError(f"HTTP {exc.code}: {detail}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"请求失败（{time.perf_counter() - started:.1f}s）：{exc}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise LLMError(f"返回缺少 choices：{json.dumps(body)[:300]}")
        text = (choices[0].get("message") or {}).get("content") or ""
        u = body.get("usage") or {}
        self.usage.add(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)),
                       self.price_per_1k)
        return text


class MockLLM(BaseLLM):
    """离线模型：确定性输出，用来验证「工程链路」而不是「模型效果」。

    说明：评测数字反映的是 评测框架的正确性与稳定性，
    换真模型后成功率会变化，但指标口径不变。
    """

    name = "mock"

    def __init__(self):
        self.usage = Usage()
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, temperature: float = 0.0,
                 max_tokens: int = 1024) -> str:
        self.calls.append((system[:40], user[:80]))
        prompt = len(system) + len(user)
        reply = self._route(system, user)
        self.usage.add(prompt // 4, len(reply) // 4)
        return reply

    def _route(self, system: str, user: str) -> str:
        if "规划器" in system:
            rules = re.findall(r"R\d+", user)
            steps = [{"tool": "extract_clauses", "args": {"doc_id": _doc_id(user)},
                      "why": "先把合同切成条款，后续按条款核查"}]
            for r in dict.fromkeys(rules):
                steps.append({"tool": "check_rule",
                              "args": {"doc_id": _doc_id(user), "rule_id": r},
                              "why": f"核查规则 {r}"})
            steps.append({"tool": "finalize", "args": {}, "why": "汇总结论"})
            return json.dumps({"goal": "核查合同风险条款", "steps": steps}, ensure_ascii=False)
        if "修复" in system:
            return json.dumps({"suggestion": "补充缺失条款并明确约定，具体见模板。"},
                              ensure_ascii=False)
        return json.dumps({"answer": "ok"}, ensure_ascii=False)


def _doc_id(text: str) -> str:
    m = re.search(r"doc_id\s*[=:]\s*([A-Za-z0-9_\-]+)", text)
    return m.group(1) if m else "doc_unknown"


def build_llm(kind: str = "mock", **kwargs) -> BaseLLM:
    if kind == "mock":
        return MockLLM()
    if kind in ("openai", "compat", "ollama"):
        return OpenAICompatLLM(**kwargs)
    raise ValueError(f"未知 llm 类型：{kind}")
