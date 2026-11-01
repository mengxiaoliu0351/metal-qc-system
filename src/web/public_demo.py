"""Public demo boundary. The private reverse proxy supplies an unexposed admin token."""
import asyncio
import base64
import hmac
import io
import json
import os
import re
import secrets
import sqlite3
import time
from contextvars import ContextVar
from contextlib import closing
from pathlib import Path

from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

PUBLIC_REQUEST = ContextVar("public_demo_request", default=False)
COOKIE = "metalqc_visitor"
MAX_BODY = 12 * 1024 * 1024


def error(status, message):
    return JSONResponse({"ok": False, "detail": message}, status_code=status)


class PublicDemoMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, admin_token, results_dir, daily_limit=60):
        super().__init__(app)
        if len(admin_token) < 32:
            raise RuntimeError("Public deployment requires a private admin token of at least 32 characters")
        self.admin_token = admin_token
        self.sessions = {}
        self.busy = 0
        self.work = asyncio.Lock()
        self.daily_limit = int(daily_limit)
        self.database = Path(results_dir) / "public_demo_usage.sqlite3"
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS usage (day TEXT, scope TEXT, n INTEGER, PRIMARY KEY(day, scope))")

    def allowance(self, address):
        # Count all public detect/chat requests, conservatively bounding paid calls.
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with closing(sqlite3.connect(self.database, timeout=5)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            for scope, limit in (("all", self.daily_limit), ("ip:" + address, 20)):
                row = db.execute("SELECT n FROM usage WHERE day=? AND scope=?", (day, scope)).fetchone()
                if row and row[0] >= limit:
                    return False
            for scope in ("all", "ip:" + address):
                db.execute("INSERT INTO usage VALUES(?,?,1) ON CONFLICT(day,scope) DO UPDATE SET n=n+1", (day, scope))
            db.execute("DELETE FROM usage WHERE day < date(?, '-7 days')", (day,))
        return True

    @staticmethod
    def validate_chat(body):
        try:
            data = json.loads(body)
            if not isinstance(data, dict):
                return False
            messages = data.get("messages", [])
            if not isinstance(messages, list) or len(messages) > 8:
                return False
            for message in messages:
                if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
                    return False
                if not isinstance(message.get("content"), str) or len(message["content"]) > 2000:
                    return False
            for key, limit in (("question", 2000), ("det_ctx", 8000)):
                if not isinstance(data.get(key, ""), str) or len(data.get(key, "")) > limit:
                    return False
            encoded = data.get("image_b64")
            if encoded:
                if not isinstance(encoded, str) or len(encoded) > 6 * 1024 * 1024:
                    return False
                raw = base64.b64decode(encoded, validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.width * image.height > 20_000_000:
                        return False
                    image.verify()
            return True
        except (ValueError, TypeError, OSError, Image.DecompressionBombError):
            return False

    async def dispatch(self, request, call_next):
        supplied = request.headers.get("x-inspection-admin", "")
        admin = hmac.compare_digest(supplied, self.admin_token)
        path, method = request.url.path, request.method
        if admin:
            # Legacy anonymous deployments may still opt out of cloud iteration.
            if method == "POST" and path == "/api/iterations" and os.getenv("CLOUD_ALLOW_TRAINING", "0") != "1":
                return error(403, "云端演示实例暂不运行训练，请在本机完成迭代评估后迁移模型；人工复核和报告仍可使用。")
            return await call_next(request)

        report_match = re.fullmatch(r"/api/report/([A-Za-z0-9_-]{1,100})", path)
        allowed = ((method in ("GET", "HEAD") and (path in ("/", "/healthz", "/api/products") or path.startswith("/static/")))
                   or (method == "GET" and report_match)
                   or (method == "POST" and path in ("/api/detect", "/api/chat")))
        if not allowed:
            return error(403, "此功能仅供管理人员使用；公开体验支持检测、AI助手和本人报告下载。")
        if method == "POST":
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return error(403, "请从本站页面提交请求。")
            try:
                if int(request.headers.get("content-length", "0")) > MAX_BODY:
                    return error(413, "上传内容过大，请将图片压缩至10MB以内。")
            except ValueError:
                return error(400, "请求长度无效。")
            body = await request.body()  # The reverse proxy enforces the streaming size limit too.
            if len(body) > MAX_BODY:
                return error(413, "上传内容过大。")
            if path == "/api/chat" and not self.validate_chat(body):
                return error(400, "对话内容或附图无效，请缩短对话或压缩图片后重试。")

        now = time.monotonic()
        self.sessions = {key: value for key, value in self.sessions.items() if now - value["time"] < 3600}
        visitor = request.cookies.get(COOKIE, "")
        if visitor not in self.sessions:
            if report_match:
                return error(403, "报告仅限检测时使用的浏览器下载，有效期为1小时。")
            if len(self.sessions) >= 512:
                return error(503, "当前体验人数较多，请稍后访问。")
            visitor = secrets.token_urlsafe(32)
            self.sessions[visitor] = {"time": now, "reports": set()}
        session = self.sessions[visitor]
        if report_match and report_match[1] not in session["reports"]:
            return error(403, "只能下载本次浏览器会话生成的报告。")

        token = PUBLIC_REQUEST.set(True)
        work = method == "POST"
        try:
            if work:
                if self.busy >= 3:
                    return error(429, "服务器正在检测，请稍后重试。")
                address = request.client.host if request.client else "unknown"
                if not self.allowance(address):
                    return error(429, "今日公开体验额度已用完，请明天再来。")
                self.busy += 1
            try:
                if work:
                    async with self.work:
                        response = await call_next(request)
                else:
                    response = await call_next(request)
            finally:
                if work:
                    self.busy -= 1
            if path == "/api/detect" and response.status_code == 200:
                content = b"".join([part async for part in response.body_iterator])
                data = json.loads(content)
                if data.get("ok") and data.get("session_id"):
                    session["reports"].add(data["session_id"])
                response = Response(content, status_code=response.status_code, headers=dict(response.headers), media_type="application/json")
            response.set_cookie(COOKIE, visitor, max_age=3600, httponly=True,
                                samesite="lax", secure=request.url.scheme == "https")
            response.headers["Cache-Control"] = "no-store" if path.startswith("/api/") else "no-cache"
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response
        finally:
            PUBLIC_REQUEST.reset(token)
