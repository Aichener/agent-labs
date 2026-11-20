"""HTTP 服务层测试：鉴权、租户隔离、入参校验、持久化与幂等。"""

from __future__ import annotations

import base64
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient      # noqa: E402

from agent import api                          # noqa: E402
from agent.store import TaskStore              # noqa: E402

KEYS = "k1:tenant-a,k2:tenant-b"
ENDPOINT = "/v1/review"
DOC = "doc_a"


def _fitz():
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    return fitz


def make_docx(paragraphs: list[str]) -> bytes:
    from io import BytesIO

    from docx import Document

    document = Document()
    for text in paragraphs:
        document.add_paragraph(text)
    buf = BytesIO()
    document.save(buf)
    return buf.getvalue()


def make_text_pdf(text: str, chinese: bool = False) -> bytes:
    fitz = _fitz()
    document = fitz.open()
    page = document.new_page()
    if chinese:
        page.insert_text((72, 100), text, fontname="china-s", fontsize=12)
    else:
        page.insert_text((72, 100), text, fontsize=12)
    data = document.tobytes()
    document.close()
    return data


def make_pdf(encrypted: bool = False) -> bytes:
    fitz = _fitz()
    document = fitz.open()
    page = document.new_page()
    if encrypted:
        page.insert_text((72, 72), "secret")
        data = document.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256,
                                user_pw="u", owner_pw="o")
    else:
        data = document.tobytes()
    document.close()
    return data


class ApiTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["AGENT_API_KEYS"] = KEYS
        db = ROOT / "logs" / "tasks_test_api.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        db.unlink(missing_ok=True)
        api._STORE = TaskStore(db)
        cls.client = TestClient(api.app)

    @classmethod
    def tearDownClass(cls):
        api._STORE.close()

    def call(self, method: str, path: str, body: dict | None = None,
             key: str | None = "k1"):
        headers = {"X-API-Key": key} if key else {}
        if method == "GET":
            response = self.client.get(path, headers=headers)
        else:
            response = self.client.post(path, json=body, headers=headers)
        return response.status_code, response.json()


class TestAuth(ApiTestBase):
    def test_healthz_without_key(self):
        status, body = self.call("GET", "/healthz", key=None)
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_missing_key_401(self):
        status, body = self.call("POST", ENDPOINT, {"doc_id": DOC}, key=None)
        self.assertEqual(status, 401)
        self.assertEqual(body["error"]["code"], "unauthorized")

    def test_wrong_key_401(self):
        status, _ = self.call("POST", ENDPOINT, {"doc_id": DOC}, key="nope")
        self.assertEqual(status, 401)


class TestTenantIsolation(ApiTestBase):
    def test_result_not_visible_across_tenants(self):
        status, body = self.call("POST", ENDPOINT, {"doc_id": DOC, "rules": ["R1"]}, key="k1")
        self.assertEqual(status, 200)
        request_id = body["request_id"]

        status, body_b = self.call("GET", f"/v1/results/{request_id}", key="k2")
        self.assertEqual(status, 403)
        self.assertEqual(body_b["error"]["code"], "forbidden")

        status, body_a = self.call("GET", f"/v1/results/{request_id}", key="k1")
        self.assertEqual(status, 200)
        self.assertEqual(body_a["verdict"], body["verdict"])


class TestValidation(ApiTestBase):
    def test_text_review_ok(self):
        status, body = self.call("POST", ENDPOINT, {
            "text": "第一条 甲方应于签署后 5 个工作日内预付合同总额的 60%。\n第二条 争议另行协商。",
            "rules": ["R1", "R2"],
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["verdict"], "fail")
        self.assertTrue(any(f["rule_id"] == "R1" for f in body["findings"]))

    def test_upload_txt_ok(self):
        content = "第一条 甲方应在签署后 10 个工作日内预付合同总额的 20%。".encode("utf-8")
        status, body = self.call("POST", ENDPOINT, {
            "filename": "合同.txt",
            "file_b64": base64.b64encode(content).decode(),
            "rules": ["R1"],
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["verdict"], "pass")

    def test_docx_upload_ok(self):
        payload = {"filename": "合同.docx",
                   "file_b64": base64.b64encode(
                       make_docx(["第一条 甲方应于签署后 10 个工作日内预付合同总额的 60%。"])).decode()}
        status, body = self.call("POST", ENDPOINT, payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["verdict"], "fail")
        self.assertIn("R1", body["failed_rules"])

    def test_pdf_with_text_ok(self):
        payload = {"filename": "report.pdf",
                   "file_b64": base64.b64encode(
                       make_text_pdf("First clause prepay sixty percent")).decode()}
        status, body = self.call("POST", ENDPOINT, payload)
        self.assertEqual(status, 200)

    def test_pdf_chinese_text_ok(self):
        payload = {"filename": "合同.pdf",
                   "file_b64": base64.b64encode(
                       make_text_pdf("第一条 甲方应于签署后十个工作日内预付合同总额的 60%。",
                                     chinese=True)).decode()}
        status, body = self.call("POST", ENDPOINT, payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["verdict"], "fail")
        self.assertIn("R1", body["failed_rules"])

    def test_pdf_without_text_422(self):
        payload = {"filename": "扫描件.pdf",
                   "file_b64": base64.b64encode(make_pdf()).decode()}
        status, body = self.call("POST", ENDPOINT, payload)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "no_text_layer")

    def test_encrypted_pdf_422(self):
        payload = {"filename": "加密.pdf",
                   "file_b64": base64.b64encode(make_pdf(encrypted=True)).decode()}
        status, body = self.call("POST", ENDPOINT, payload)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "encrypted_file")

    def test_unsupported_suffix_415(self):
        status, body = self.call("POST", ENDPOINT, {
            "filename": "合同.xlsx", "file_b64": base64.b64encode(b"x").decode()})
        self.assertEqual(status, 415)
        self.assertEqual(body["error"]["code"], "unsupported_type")

    def test_bad_base64_400(self):
        status, body = self.call("POST", ENDPOINT, {
            "filename": "a.txt", "file_b64": "!!!not-base64!!!"})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "bad_base64")

    def test_too_many_pages_422(self):
        status, body = self.call("POST", ENDPOINT, {"text": "甲" * (201 * 1500)})
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "too_many_pages")

    def test_too_large_413(self):
        status, body = self.call("POST", ENDPOINT, {"text": "甲" * (6 * 1024 * 1024)})
        self.assertEqual(status, 413)
        self.assertEqual(body["error"]["code"], "too_large")

    def test_empty_text_422(self):
        status, body = self.call("POST", ENDPOINT, {"text": "   "})
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "empty_content")

    def test_unknown_doc_404(self):
        status, body = self.call("POST", ENDPOINT, {"doc_id": "nope"})
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "doc_not_found")

    def test_bad_rules_422(self):
        status, body = self.call("POST", ENDPOINT, {"doc_id": DOC, "rules": ["R99"]})
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "bad_rules")

    def test_bad_json_400(self):
        response = self.client.post(ENDPOINT, content=b"{not json",
                                    headers={"X-API-Key": "k1",
                                             "Content-Type": "application/json"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "bad_json")


class TestTimeout(ApiTestBase):
    def test_timeout_504(self):
        original = api.TIMEOUT_S
        api.TIMEOUT_S = 0.001
        try:
            status, body = self.call("POST", ENDPOINT, {
                "text": "第一条 甲方应预付 60%。", "rules": ["R1"]})
        finally:
            api.TIMEOUT_S = original
        self.assertIn(status, (200, 504))
        if status == 504:
            self.assertEqual(body["error"]["code"], "timeout")


class TestPersistence(ApiTestBase):
    """P0-4：同参数不重复跑，重启后结果还在。"""

    def setUp(self):
        api._store().clear()

    def test_same_payload_replays(self):
        payload = {"doc_id": DOC}
        s1, b1 = self.call("POST", ENDPOINT, payload, key="k1")
        s2, b2 = self.call("POST", ENDPOINT, payload, key="k1")
        self.assertEqual((s1, s2), (200, 200))
        self.assertFalse(b1["replayed"])
        self.assertTrue(b2["replayed"])
        self.assertEqual(b1["request_id"], b2["request_id"])

    def test_same_payload_different_tenant_not_shared(self):
        payload = {"doc_id": DOC}
        _, b1 = self.call("POST", ENDPOINT, payload, key="k1")
        _, b2 = self.call("POST", ENDPOINT, payload, key="k2")
        self.assertFalse(b1["replayed"])
        self.assertFalse(b2["replayed"])
        self.assertNotEqual(b1["request_id"], b2["request_id"])

    def test_result_survives_store_reopen(self):
        _, body = self.call("POST", ENDPOINT, {"doc_id": DOC}, key="k1")
        request_id = body["request_id"]
        path = api._STORE.path
        api._STORE.close()
        api._STORE = TaskStore(path)          # 模拟进程重启，重新打开同一个库
        status, again = self.call("GET", f"/v1/results/{request_id}", key="k1")
        self.assertEqual(status, 200)
        self.assertEqual(again["verdict"], body["verdict"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
