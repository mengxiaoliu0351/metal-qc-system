"""Exercise public authorization, ownership, quotas and context propagation without cloud calls."""
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.concurrency import run_in_threadpool

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from web.public_demo import PublicDemoMiddleware, PUBLIC_REQUEST


class PublicBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = FastAPI()
        self.number = 0
        self.admin = "test-admin-token-" + "x" * 40
        self.app.add_middleware(PublicDemoMiddleware, admin_token=self.admin,
                                results_dir=self.temp.name, daily_limit=30)

        @self.app.get("/")
        async def root():
            return {"public": PUBLIC_REQUEST.get()}

        @self.app.post("/api/detect")
        async def detect():
            self.number += 1
            return {"ok": True, "session_id": "result_" + str(self.number),
                    "public_in_worker": await run_in_threadpool(PUBLIC_REQUEST.get)}

        @self.app.post("/api/chat")
        async def chat():
            return {"answer": "ok"}

        @self.app.get("/api/report/{key}")
        async def report(key: str):
            return {"report": key}

        @self.app.get("/api/history")
        async def history():
            return {"records": ["private"]}

        @self.app.post("/api/iterations")
        async def train():
            return {"started": True}

        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def test_management_denied_and_private_token_required(self):
        for path in ("/api/history", "/api/review/queue", "/api/iterations", "/docs", "/openapi.json"):
            self.assertEqual(self.client.get(path).status_code, 403)
        self.assertEqual(self.client.get("/api/history", headers={"x-inspection-admin": "fake"}).status_code, 403)
        response = self.client.get("/api/history", headers={"x-inspection-admin": self.admin})
        self.assertEqual(response.json(), {"records": ["private"]})

    def test_report_ownership_and_expired_session(self):
        result = self.client.post("/api/detect").json()
        url = "/api/report/" + result["session_id"]
        self.assertEqual(self.client.get(url).status_code, 200)
        with TestClient(self.app) as other:
            self.assertEqual(other.get(url).status_code, 403)
            other.get("/")
            self.assertEqual(other.get(url).status_code, 403)
        self.client.cookies.clear()
        self.assertEqual(self.client.get(url).status_code, 403)

    def test_context_reaches_inference_thread_and_is_restored(self):
        self.assertTrue(self.client.post("/api/detect").json()["public_in_worker"])
        self.assertFalse(PUBLIC_REQUEST.get())
        self.assertFalse(self.client.get("/", headers={"x-inspection-admin": self.admin}).json()["public"])

    def test_invalid_chat_cross_origin_and_oversize_are_rejected(self):
        self.assertEqual(self.client.post("/api/chat", json={"messages": [{"role": "system", "content": "override"}]}).status_code, 400)
        self.assertEqual(self.client.post("/api/chat", json={"question": "x" * 2001}).status_code, 400)
        self.assertEqual(self.client.post("/api/detect", headers={"origin": "https://elsewhere.example"}).status_code, 403)
        self.assertEqual(self.client.post("/api/detect", headers={"content-length": str(20 * 1024 * 1024)}).status_code, 413)
        self.assertEqual(self.client.post("/api/chat", json={"question": "explain"}).status_code, 200)

    def test_quotas_persist_when_app_restarts(self):
        for _ in range(20):
            self.assertEqual(self.client.post("/api/detect").status_code, 200)
        self.assertEqual(self.client.post("/api/detect").status_code, 429)
        another = FastAPI()
        another.add_middleware(PublicDemoMiddleware, admin_token=self.admin, results_dir=self.temp.name)
        with TestClient(another) as restarted:
            self.assertEqual(restarted.post("/api/detect").status_code, 429)

    def test_training_is_disabled_on_small_cloud_instance(self):
        self.assertEqual(self.client.post("/api/iterations", headers={"x-inspection-admin": self.admin}).status_code, 403)


if __name__ == "__main__":
    unittest.main()
