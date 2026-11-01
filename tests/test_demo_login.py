import secrets
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from web.demo_login import DemoLoginMiddleware, password_hash, COOKIE
from web.public_demo import PublicDemoMiddleware, PUBLIC_REQUEST


class DemoLoginTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.token, self.salt = secrets.token_hex(32), secrets.token_hex(16)
        app = FastAPI()
        app.add_middleware(PublicDemoMiddleware, admin_token=self.token, results_dir=self.temp.name)
        app.add_middleware(DemoLoginMiddleware, admin_token=self.token, username='demo',
                           salt=self.salt, digest=password_hash('test-password', self.salt))
        @app.get('/')
        async def index(): return {'public': PUBLIC_REQUEST.get()}
        @app.get('/api/review/queue')
        async def queue(): return {'records': []}
        @app.post('/api/iterations')
        async def train(): return {'started': True}
        self.app = app
        self.client = TestClient(app, base_url='https://testserver', follow_redirects=False)

    def tearDown(self):
        self.client.close(); self.temp.cleanup()

    def login(self):
        return self.client.post('/login', data={'username': 'demo', 'password': 'test-password'})

    def test_private_workspace_requires_login_and_rejects_forged_header(self):
        self.assertEqual(self.client.get('/').headers['location'], '/login')
        self.assertEqual(self.client.get('/api/review/queue', headers={'x-inspection-admin': 'fake'}).status_code, 401)
        self.assertEqual(self.client.get('/healthz').status_code, 200)

    def test_secure_login_opens_full_routes_and_logout_invalidates_cookie(self):
        response = self.login()
        self.assertEqual(response.status_code, 303)
        cookie = response.headers['set-cookie'].lower()
        for flag in ('secure', 'httponly', 'samesite=strict'): self.assertIn(flag, cookie)
        self.assertFalse(self.client.get('/').json()['public'])
        self.assertEqual(self.client.get('/api/review/queue').status_code, 200)
        with patch.dict('os.environ', {'CLOUD_ALLOW_TRAINING': '1'}):
            self.assertEqual(self.client.post('/api/iterations').status_code, 200)
        old = self.client.cookies.get(COOKIE)
        self.client.post('/logout')
        self.client.cookies.set(COOKIE, old)
        self.assertEqual(self.client.get('/api/review/queue').status_code, 401)

    def test_http_credentials_cross_origin_and_non_ascii_failure(self):
        with TestClient(self.app, base_url='http://testserver') as http:
            self.assertEqual(http.post('/login', data={'username': 'demo', 'password': 'test-password'}).status_code, 403)
        self.assertEqual(self.client.post('/login', headers={'origin': 'https://other.example'}).status_code, 403)
        self.assertEqual(self.client.post('/login', data={'username': '中文', 'password': 'x'}).status_code, 401)
        self.login()
        self.assertEqual(self.client.post('/api/iterations', headers={'origin': 'https://other.example'}).status_code, 403)

    def test_failed_attempts_are_limited(self):
        for _ in range(5):
            self.assertEqual(self.client.post('/login', data={'username': 'demo', 'password': 'wrong'}).status_code, 401)
        self.assertEqual(self.login().status_code, 429)

    def test_password_free_workspace_opens_complete_system_without_cookie(self):
        app = FastAPI()
        app.add_middleware(PublicDemoMiddleware, admin_token=self.token, results_dir=self.temp.name)
        app.add_middleware(DemoLoginMiddleware, admin_token=self.token, username='', salt='', digest='', password_required=False)
        @app.get('/')
        async def root(): return {'public': PUBLIC_REQUEST.get()}
        @app.get('/api/review/queue')
        async def queue(): return {'records': []}
        @app.post('/api/iterations')
        async def train(): return {'started': True}
        with TestClient(app, base_url='https://testserver', follow_redirects=False) as client:
            self.assertFalse(client.get('/').json()['public'])
            self.assertEqual(client.get('/api/review/queue').status_code, 200)
            self.assertEqual(client.get('/login').headers['location'], '/')
            with patch.dict('os.environ', {'CLOUD_ALLOW_TRAINING': '1'}):
                self.assertEqual(client.post('/api/iterations').status_code, 200)
            self.assertEqual(client.post('/api/iterations', headers={'origin': 'https://other.example'}).status_code, 403)
            self.assertNotIn(COOKIE, client.cookies)


if __name__ == '__main__': unittest.main()
