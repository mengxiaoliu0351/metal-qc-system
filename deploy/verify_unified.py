"""External HTTPS acceptance checks. Credentials are read locally and never logged."""
import json
import os
import time
from pathlib import Path
import requests

root = Path(__file__).resolve().parents[1]
output = root / '.preview/aliyun-deploy-20261025/verification'
base = 'https://8.210.4.196'
client = requests.Session(); client.trust_env = False
checks = {}; detections = []
def check(name, condition):
    checks[name] = bool(condition)
    print(name, 'PASS' if condition else 'FAIL', flush=True)

check('http_redirect', client.get('http://8.210.4.196/', allow_redirects=False, timeout=15).headers.get('location', '').startswith(base))
password_required = os.getenv('VERIFY_LOGIN') == '1'
if password_required:
    lines = (output.parent / '演示登录信息.txt').read_text(encoding='utf-8').splitlines()
    username = next(x.split('：', 1)[1] for x in lines if x.startswith('演示账号：'))
    password = next(x.split('：', 1)[1] for x in lines if x.startswith('演示密码：'))
    check('anonymous_api_denied', client.get(base + '/api/history', timeout=15).status_code == 401)
    login = client.post(base + '/login', data={'username':username, 'password':password}, allow_redirects=False, timeout=30)
    check('login_success', login.status_code == 303)
    check('secure_cookie', all(x in login.headers.get('set-cookie','').lower() for x in ('secure','httponly','samesite=strict')))
else:
    check('homepage_direct', client.get(base + '/', allow_redirects=False, timeout=20).status_code == 200)
    check('legacy_login_redirect', client.get(base + '/login', allow_redirects=False, timeout=15).headers.get('location') == '/')
page = client.get(base + '/', timeout=20).text
check('complete_workspace', 'public-demo.js' not in page and 'cloud-admin.js' not in page)
check('session_controls', ('session-ui.js' in page) == password_required)
check('readable_styles', 'clarity.css' in page)
for path in ('/api/products', '/api/history', '/api/review/queue', '/api/iterations'):
    check(path, client.get(base + path, timeout=20).status_code == 200)
# The owner may already be using the live system; preserve those labels and candidates.
summary = client.get(base + '/api/review/queue?product=metal_nut', timeout=20).json()['summary']
if summary['iteration_approved']:
    runs = client.get(base + '/api/iterations', timeout=20).json()['runs']
    check('training_enabled_with_label_gate', any(x.get('product') == 'metal_nut' and x.get('status') == 'completed' for x in runs))
else:
    train = client.post(base + '/api/iterations', json={'product':'metal_nut','mode':'model'}, timeout=20)
    check('training_enabled_with_label_gate', train.status_code == 400 and '复核' in train.json().get('detail',''))
for name, route in [('nut-good-012','low_risk'),('nut-scratch-001','high_risk'),('nut-scratch-020','vlm_review')]:
    path = root / 'src/web/static/samples' / (name + '.png')
    started = time.monotonic()
    with path.open('rb') as image:
        response = client.post(base + '/api/detect', files={'file':(path.name,image,'image/png')}, data={'product':'metal_nut'}, timeout=180)
    data = response.json()
    check(name, response.status_code == 200 and data.get('ok') and data.get('route') == route)
    detections.append({'sample':name,'score':data.get('score'),'route':data.get('route'),'seconds':round(time.monotonic()-started,2)})
    if route == 'vlm_review': check('gray_vlm', data.get('cognition',{}).get('ok') is True)
    pdf = client.get(base + '/api/report/' + data['session_id'], timeout=60)
    check(name + '_report', pdf.status_code == 200 and pdf.content.startswith(b'%PDF'))
    (output / (name + '-unified.pdf')).write_bytes(pdf.content)
chat = client.post(base + '/api/chat', json={'question':'请用一句话说明人工复核在质检中的作用。'}, timeout=180).json()
answer = chat.get('answer','')
check('ai_assistant', bool(answer) and not any(x in answer for x in ('暂时不可用','请求失败','请求未完成')))
check('logout', client.post(base+'/logout', allow_redirects=False, timeout=15).status_code == 303)
check('access_after_logout', client.get(base+'/api/history', timeout=15).status_code == (401 if password_required else 200))
(output/'unified-smoke.json').write_text(json.dumps({'checks':checks,'detections':detections},ensure_ascii=False,indent=2),encoding='utf-8')
assert all(checks.values()), checks
