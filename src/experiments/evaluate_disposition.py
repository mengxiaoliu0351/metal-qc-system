"""Replay frozen features through disposition rules without VLM calls or history writes."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from gating.router import gate
from perception.feature_backends import binary_probability
from review.disposition import InspectionPolicy, image_quality
from review.workflow import read, write, image_hash


def evaluate(root=ROOT):
    results = root / 'results'
    policy = InspectionPolicy(root, results)
    products = read(results / 'perception/banks/products.json')
    rows = []
    for config in products:
        product = config['key']
        profile = policy.profile(config)
        base = results / 'experiments/perception_upgrade_v1'
        split = read(base / (product + '_split.json'))
        with np.load(base / (product + '_features.npz'), allow_pickle=False) as z: data = {k: z[k].copy() for k in z.files}
        meta_path = results / 'perception/optimized' / config['perception_artifact']
        meta = read(meta_path)
        with np.load(meta_path.parent / meta['arrays'], allow_pickle=False) as z: params = {k: z[k].copy() for k in z.files}
        probability = binary_probability(data['global_features'], params, 'global')
        if meta['mode'] == 'fast_fusion': probability = binary_probability(np.c_[data['clip_scores'], probability], params, 'fusion')
        index = {str(p).replace('\\', '/'): i for i, p in enumerate(data['paths'])}
        cases = []
        for relative in split['test']:
            relative = relative.replace('\\', '/')
            path = root / relative; i = index[relative]; score = float(probability[i])
            route = gate(score, t_low=config['threshold'], t_high=config['t_high'])
            with Image.open(path) as image: quality = image_quality(image)
            rec = {'image_hash': image_hash(path), 'quality_check': quality, 'score': score, 'gate': route,
                   'cognition': {'ok': True}}
            action = policy.decide(rec, config, data['global_features'][i])
            cases.append({'path': relative, 'truth': 'normal' if path.parent.name == 'good' else 'defect',
                          'score': score, 'route': route['route'], 'status': action['status'],
                          'requires_review': action['requires_review'], 'auto_result': action['auto_result'],
                          'reason_code': action.get('reason_code'), 'sampled': action['sampled']})
        automatic = [c for c in cases if c['auto_result']]
        rows.append({'product': product, 'n': len(cases), 'normal_n': sum(c['truth'] == 'normal' for c in cases),
                     'automatic_decisions': len(automatic), 'audit_selected': sum(c['sampled'] for c in cases),
                     'human_tasks': sum(c['requires_review'] for c in cases),
                     'automatic_errors': sum(c['auto_result'] != c['truth'] for c in automatic),
                     'counts': dict(Counter(c['status'] for c in cases)),
                     'model_key': profile['model_key'], 'auto_low': profile['auto_low'], 'auto_high': profile['auto_high'],
                     'allow_normal': profile['allow_normal'], 'allow_isolate': profile['allow_isolate'], 'cases': cases})
    return {'scope': '冻结回归重放，已重复用于版本评估，不是新产线盲测。灰区未调用VLM，仍转人工。人工任务数包含稳定随机抽查。',
            'settings': policy.public_settings(), 'products': rows}


if __name__ == '__main__':
    result = evaluate()
    target = ROOT / 'results/experiments/selective_inspection_20261026.json'
    write(target, result)
    print(json.dumps([{k: v for k, v in p.items() if k != 'cases'} for p in result['products']], ensure_ascii=False, indent=2))
