"""One authenticated demonstration workspace. Cookies hold random session IDs only."""
import hashlib
import hmac
import html
import os
import secrets
import time
from urllib.parse import parse_qs

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE = "metalqc_demo_login"


def password_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 260000).hex()


def login_page(message=""):
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>演示登录 · MetalQC</title>
<style>*{box-sizing:border-box}body{margin:0;background:#edf2f8;color:#193453;font:16px/1.7 "Microsoft YaHei",sans-serif}main{min-height:100vh;display:grid;grid-template-columns:1.1fr 1fr}.identity{padding:10vh 7vw;background:#11243c;color:#e8f1fb;display:flex;flex-direction:column;justify-content:center;background-image:linear-gradient(#8bb6df09 1px,transparent 1px),linear-gradient(90deg,#8bb6df09 1px,transparent 1px);background-size:44px 44px}.brand{font-size:21px;letter-spacing:3px;color:#88c5ff}.identity h1{font-size:clamp(32px,4vw,54px);line-height:1.4;margin:55px 0 24px}.identity p{color:#b8cce3;max-width:450px}.line{height:4px;width:64px;background:#53a8ff;margin:20px 0}.steps{padding:0;list-style:none;margin-top:36px;display:flex;gap:28px;color:#cee0f3;font-size:14px}.login{display:flex;align-items:center;justify-content:center;padding:40px}.card{width:100%;max-width:430px}small{color:#526d8c;font-size:13px}h2{font-size:30px;margin:10px 0}label{display:block;margin:23px 0 8px;font-weight:600}input{width:100%;font:inherit;border:1px solid #aabed6;border-radius:4px;padding:12px 14px;background:white;min-height:50px}input:focus{outline:3px solid #9bc6ff;border-color:#2366cb}button{width:100%;font:inherit;font-size:16px;font-weight:600;min-height:52px;margin-top:28px;color:white;border:0;border-radius:4px;background:#1559d6;cursor:pointer}button:hover{background:#1048af}.error{color:#a42e35;background:#ffe9eb;padding:10px 14px;border-radius:4px}.hint{font-size:13px;color:#526c89;margin-top:22px}.footer{font-size:12px;color:#7890ab;margin-top:40px}@media(max-width:760px){main{display:block}.identity{padding:30px}.identity h1{font-size:30px;margin:24px 0 10px}.steps{margin-top:20px}.login{padding:32px 24px}}</style>
<main><section class="identity"><div class="brand">METALQC / 金属表面检验</div><div class="line"></div><h1>从发现异常<br>到确认每一次判断。</h1><p>本地感知、灰区AI复核与标准检索，接入同一个完整检验流程。</p><ul class="steps"><li>01 工件检测</li><li>02 人工核验</li><li>03 迭代评估</li></ul></section><section class="login"><div class="card"><small>INSPECTION WORKSPACE</small><h2>进入演示工作台</h2><p>登录后可使用检测、AI助手、人工复核、报告和模型迭代。</p>''' + (f'<p class="error" role="alert">{html.escape(message)}</p>' if message else '') + '''<form action="/login" method="post"><label for="username">演示账号</label><input id="username" name="username" autocomplete="username" required maxlength="80"><label for="password">演示密码</label><input id="password" name="password" type="password" autocomplete="current-password" required maxlength="200"><button type="submit">登录并进入完整系统 →</button></form><p class="hint">请使用项目负责人提供的演示账号。所有AI结果均为辅助意见，最终质量结论由人工核验。</p><p class="footer">双链路智能质检辅助系统 · 人工智能应用实践</p></div></section></main></html>'''


class DemoLoginMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, admin_token, username, salt, digest, require_https=True, password_required=True):
        super().__init__(app)
        if len(admin_token) < 32 or (password_required and (not username or len(salt) < 32 or len(digest) != 64)):
            raise RuntimeError("Missing demonstration login configuration")
        self.admin_token, self.username, self.salt, self.digest = admin_token, username, salt, digest
        self.require_https = require_https
        self.password_required = password_required
        self.sessions, self.failures = {}, {}
        self.busy = 0

    async def dispatch(self, request, call_next):
        # The loopback-only SSH proxy remains a recovery/maintenance entry.
        if hmac.compare_digest(request.headers.get("x-inspection-admin", ""), self.admin_token):
            return await call_next(request)
        path = request.url.path
        if path == "/healthz" and request.method == "GET":
            return JSONResponse({"ok": True, "service": "metalqc"})
        now = time.monotonic()
        self.sessions = {key: value for key, value in self.sessions.items() if value > now}
        self.failures = {key: value for key, value in self.failures.items() if now - value[0] < 300}
        secure = request.url.scheme == "https"
        cookie = request.cookies.get(COOKIE, "")
        authenticated = not self.password_required or cookie in self.sessions
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "请从本站页面提交请求"}, status_code=403)
        if path == "/logout" and request.method == "POST":
            self.sessions.pop(cookie, None)
            response = RedirectResponse("/login" if self.password_required else "/", status_code=303)
            response.delete_cookie(COOKIE)
            return response
        if path == "/login":
            if not self.password_required:
                return RedirectResponse("/", status_code=303)
            if self.require_https and not secure:
                return HTMLResponse(login_page("请通过HTTPS地址登录"), status_code=403)
            if request.method == "GET":
                return RedirectResponse("/", status_code=303) if authenticated else HTMLResponse(login_page(), headers={"Cache-Control": "no-store"})
            if request.method != "POST":
                return JSONResponse({"detail": "不支持的操作"}, status_code=405)
            address = request.client.host if request.client else "unknown"
            started, count = self.failures.get(address, (now, 0))
            if count >= 5:
                return HTMLResponse(login_page("尝试次数较多，请5分钟后重试"), status_code=429)
            body = await request.body()
            if len(body) > 4096:
                return JSONResponse({"detail": "登录内容过长"}, status_code=413)
            values = parse_qs(body.decode("utf-8", errors="replace"))
            username = values.get("username", [""])[0]
            password = values.get("password", [""])[0]
            valid = hmac.compare_digest(username.encode(), self.username.encode()) and hmac.compare_digest(password_hash(password, self.salt), self.digest)
            if not valid:
                self.failures[address] = (started, count + 1)
                return HTMLResponse(login_page("账号或密码不正确"), status_code=401)
            self.failures.pop(address, None)
            if len(self.sessions) >= 128:
                return HTMLResponse(login_page("当前演示会话较多，请稍后登录"), status_code=503)
            session = secrets.token_urlsafe(32)
            self.sessions[session] = now + 8 * 3600
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(COOKIE, session, httponly=True, secure=secure, samesite="strict", max_age=8 * 3600)
            response.headers["Cache-Control"] = "no-store"
            return response
        if not authenticated:
            if path.startswith("/api/"):
                return JSONResponse({"ok": False, "detail": "演示登录已失效，请刷新页面重新登录"}, status_code=401)
            return RedirectResponse("/login", status_code=303)
        request.scope["headers"] = [(k, v) for k, v in request.scope["headers"] if k.lower() != b"x-inspection-admin"] + [(b"x-inspection-admin", self.admin_token.encode())]
        request.scope["demo_authenticated"] = self.password_required
        work = request.method == "POST" and path in ("/api/detect", "/api/chat")
        if work and self.busy >= 3:
            return JSONResponse({"detail": "当前检测任务较多，请稍后再试"}, status_code=429)
        if work:
            self.busy += 1
        try:
            response = await call_next(request)
        finally:
            if work:
                self.busy -= 1
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        return response
