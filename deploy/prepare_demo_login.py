"""Run on the server. Preserve provider/admin secrets and create demo credentials once."""
import hashlib
import os
import secrets
from pathlib import Path

root = Path('/opt/metalqc')
env = root / 'runtime.env'
values = dict(line.split('=', 1) for line in env.read_text().splitlines() if '=' in line)
if not all(values.get(k) for k in ('DEMO_USERNAME', 'DEMO_PASSWORD_SALT', 'DEMO_PASSWORD_HASH')):
    password = secrets.token_urlsafe(12)
    salt = secrets.token_hex(16)
    values.update(DEMO_USERNAME='demo', DEMO_PASSWORD_SALT=salt,
                  DEMO_PASSWORD_HASH=hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), 260000).hex())
    credential = root / 'demo-login.txt'
    credential.write_text('完整系统：https://8.210.4.196/\n演示账号：demo\n演示密码：' + password +
                          '\n\n共享演示账号具有人工复核和模型发布权限，仅分享给答辩参与者。\n', encoding='utf-8')
    os.chmod(credential, 0o600)
env.write_text(''.join(k + '=' + v + '\n' for k, v in values.items()))
os.chmod(env, 0o600)
print('Demo login configured; existing API and administrator secrets preserved.')
