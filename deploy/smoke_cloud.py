"""Run inside the app container through stdin after the proxy is healthy."""
import json
import time
import sys
from pathlib import Path

import requests
sys.path.insert(0, '/app/src')

public = 'http://proxy:8080'
private = 'http://proxy:8081'
client = requests.Session()
receipt = {'checks': {}, 'detections': []}
out = Path('/app/results/deployment_checks')
out.mkdir(exist_ok=True)

def check(name, value):
    receipt['checks'][name] = bool(value)
    print(name, 'PASS' if value else 'FAIL', flush=True)

check('health', client.get(public + '/healthz', timeout=10).json().get('ok'))
check('products', len(client.get(public + '/api/products', timeout=10).json()['products']) == 3)
check('public_history_denied', client.get(public + '/api/history', timeout=10).status_code == 403)
check('public_header_spoof_denied', client.get(public + '/api/history', headers={'X-Inspection-Admin': 'fake'}, timeout=10).status_code == 403)

for sample, expected in [('nut-good-012', 'low_risk'), ('nut-scratch-001', 'high_risk'), ('nut-scratch-020', 'vlm_review')]:
    path = Path('/app/src/web/static/samples') / (sample + '.png')
    start = time.monotonic()
    with path.open('rb') as image:
        response = client.post(public + '/api/detect', files={'file': (path.name, image, 'image/png')},
                               data={'product': 'metal_nut'}, timeout=180)
    data = response.json()
    item = {'sample': sample, 'status': response.status_code, 'ok': data.get('ok'),
            'score': data.get('score'), 'route': data.get('route'),
            'seconds': round(time.monotonic() - start, 2), 'session_id': data.get('session_id'),
            'cognition_ok': (data.get('cognition') or {}).get('ok'), 'references': len(data.get('references', []))}
    receipt['detections'].append(item)
    print(json.dumps(item), flush=True)
    check(sample + '_route', data.get('ok') and data.get('route') == expected)
    if sample == 'nut-scratch-020':
        check('gray_real_vlm', (data.get('cognition') or {}).get('ok') is True)
    if data.get('session_id'):
        report_url = '/api/report/' + data['session_id']
        pdf = client.get(public + report_url, timeout=60)
        check(sample + '_pdf', pdf.status_code == 200 and pdf.content.startswith(b'%PDF'))
        if pdf.content.startswith(b'%PDF'):
            (out / (sample + '.pdf')).write_bytes(pdf.content)
        check(sample + '_pdf_private', requests.get(public + report_url, timeout=10).status_code == 403)

start = time.monotonic()
chat = client.post(public + '/api/chat', json={'question': '请用两句话说明金属表面质检中的人工复核有什么作用。'}, timeout=180).json()
answer = chat.get('answer', '')
check('real_chat', bool(answer) and not any(word in answer for word in ('暂时不可用', '请求未完成', '请求失败')))
receipt['chat'] = {'seconds': round(time.monotonic() - start, 2), 'characters': len(answer)}
queue = requests.get(private + '/api/review/queue?limit=5', timeout=15)
check('private_review', queue.status_code == 200)
check('private_training_disabled', requests.post(private + '/api/iterations', json={'product': 'metal_nut'}, timeout=15).status_code == 403)
from interface import report
receipt['pdf_font'] = report._FONT_NAME
check('chinese_pdf_font', report._FONT_NAME == 'CN')
(out / 'cloud-smoke.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print('ALL_CHECKS_PASS', all(receipt['checks'].values()), flush=True)
