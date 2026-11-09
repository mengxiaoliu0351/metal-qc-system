"""One frozen nonlinear-head probe, cached development only before veto checks.

All evaluation cohorts have previously been viewed. Passing historical gates
never grants production or automatic archival permission.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from experiments import m2_body_robust as previous
from experiments.m2_body_quality import load_features, measure, save_arrays
from experiments.prepare_m2_body import sha
from perception.body_kernel import export_kernel, kernel_scores
from perception.body_quality import BodyQualityModel, body_scores
from products.families import read, write

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'results/experiments/m2_body_quality_v5_kernel'
PROTECTED = ['results/perception/banks/products.json', 'results/perception/family_active.json',
             'results/product_families/screw/body_quality_candidate.json',
             'results/product_families/screw/operational_router.json',
             'results/product_families/screw/appearance_active.json',
             'results/data_intake/known_image_derivatives.json',
             'results/evaluation_reservations.json', 'results/inspection_policy.json']


def prepare(output):
    if (output/'protocol.json').exists(): raise ValueError('Protocol already frozen')
    parent = read(previous.OUTPUT/'protocol.json')
    samples = parent['samples']
    roles = {}
    for s in samples: roles.setdefault(s['pixel_hash'], set()).add(s['role'])
    if any(len(v) != 1 for v in roles.values()): raise ValueError('Cross-partition image')
    reservations = read(ROOT/'results/evaluation_reservations.json').get('global', {})
    if any(s['pixel_hash'] in reservations for s in samples if s['role'] == 'train'):
        raise ValueError('Frozen evaluation image entered training')
    features = {}
    for s in samples:
        for variant in previous.VARIANTS if s['role'] == 'train' else ('original',):
            path = previous.feature_path(previous.OUTPUT, s, variant)
            features[path.name] = sha(path)
    protocol = {'version': 'm2-body-quality-v5-kernel', 'samples': samples,
                'parent_protocol_sha256': sha(previous.OUTPUT/'protocol.json'),
                'parent_arrays_sha256': sha(previous.OUTPUT/'quality.npz'),
                'implementation_sha256': sha(Path(__file__)),
                'predictor_sha256': sha(ROOT/'src/perception/body_kernel.py'),
                'protected': {name: sha(ROOT/name) for name in PROTECTED},
                'feature_sha256': features, 'classifier': {'kernel': 'rbf', 'C': .1,
                'gamma': 'scale', 'class_weight': 'balanced', 'probability': False},
                'training_policy': 'same V2 photometric views; each source image total weight one',
                'seed': 20261108, 'candidate_count': 1,
                'calibration_target': parent['calibration_target'],
                'quality_gate': parent['historical_regression_gate'],
                'stop_rule': 'stop at failed development or qualification; at most one final historical replay; no retuning',
                'evaluation_scope': 'all 200/400/998 and HEAD cohorts are historical; not new blind tests',
                'release_permission': False, 'automatic_archival': False}
    output.mkdir(parents=True, exist_ok=True)
    write(output/'protocol.json', protocol)
    print({'frozen': protocol['version'], 'roles': dict(Counter(s['role'] for s in samples)),
           'candidates': 1, 'new_extraction': False}, flush=True)


def verify(output):
    p = read(output/'protocol.json')
    if sha(Path(__file__)) != p['implementation_sha256'] or sha(ROOT/'src/perception/body_kernel.py') != p['predictor_sha256']:
        raise ValueError('Frozen implementation changed')
    if sha(previous.OUTPUT/'protocol.json') != p['parent_protocol_sha256'] or sha(previous.OUTPUT/'quality.npz') != p['parent_arrays_sha256']:
        raise ValueError('Parent artifact changed')
    if any(sha(ROOT/name) != expected for name,expected in p['protected'].items()):
        raise ValueError('Live protected state changed')
    return p


def vectors(output, rows, variant='original'):
    manifest = read(output/'protocol.json')['feature_sha256']
    values = []
    for s in rows:
        path = previous.feature_path(previous.OUTPUT, s, variant)
        if sha(path) != manifest[path.name]: raise ValueError('Cached feature changed')
        with np.load(path, allow_pickle=False) as z: values.append(z['vector'])
    return np.asarray(values, dtype=float)


def labels(rows): return np.array([s['label'] == 'defect' for s in rows], dtype=int)


def fit(output):
    if (output/'candidate.json').exists(): raise ValueError('Do not refit frozen candidate')
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from threadpoolctl import threadpool_limits
    p = verify(output)
    train = [s for s in p['samples'] if s['role'] == 'train']
    calibration = [s for s in p['samples'] if s['role'] == 'calibration']
    x = np.concatenate([vectors(output, train, v) for v in previous.VARIANTS])
    y = np.tile(labels(train), len(previous.VARIANTS))
    pipeline = Pipeline([('sc', StandardScaler()), ('clf', SVC(**p['classifier'], random_state=p['seed']))])
    with threadpool_limits(limits=2):
        pipeline.fit(x, y, clf__sample_weight=np.full(len(y), 1/len(previous.VARIANTS)))
    params = export_kernel(pipeline)
    xc = vectors(output, calibration)
    scores = kernel_scores(xc, params)
    # Export parity protects a future online consumer against silent score changes.
    reference = 1/(1+np.exp(-np.clip(pipeline.decision_function(xc), -700, 700)))
    if not np.allclose(reference, scores, atol=1e-10): raise ValueError('Portable predictor mismatch')
    low,high,feasible = previous.conservative_threshold(labels(calibration), scores, p['calibration_target'])
    metric = measure(labels(calibration), scores, low, high)
    save_arrays(output/'candidate.npz', **params)
    write(output/'candidate.json', {'version': p['version'], 'arrays': 'candidate.npz',
          'arrays_sha256': sha(output/'candidate.npz'), 'protocol_sha256': sha(output/'protocol.json'),
          'thresholds': [low,high], 'development_target_met': feasible,
          'automatic_archival': False, 'release_permission': False})
    write(output/'development_report.json', {'calibration': metric, 'development_target_met': feasible,
          'train_images': len(train), 'calibration_images': len(calibration),
          'support_vectors': len(params['support']), 'heldout_features_read': False,
          'portable_parity_max_error': float(np.max(np.abs(reference-scores)))})
    print({'calibration': metric, 'development_target_met': feasible}, flush=True)


def candidate(output):
    p = verify(output); meta = read(output/'candidate.json')
    if sha(output/meta['arrays']) != meta['arrays_sha256'] or sha(output/'protocol.json') != meta['protocol_sha256']:
        raise ValueError('Unpaired candidate')
    with np.load(output/meta['arrays'], allow_pickle=False) as z: params = {k:z[k].copy() for k in z.files}
    return p,meta,params


def cohort(output, rows, meta, params):
    x = vectors(output, rows); scores = kernel_scores(x, params)
    baseline = BodyQualityModel(previous.OUTPUT/'quality.json')
    xb,raw = load_features(previous.OUTPUT, rows)
    old = body_scores(xb, raw, baseline.params, baseline.meta['selected_mode'])
    coverage = float(np.mean([baseline.predict(v,r)['scope_ok'] for v,r in zip(xb,raw)]))
    return {'candidate': measure(labels(rows), scores, *meta['thresholds']),
            'v2_fixed': measure(labels(rows), old, *baseline.meta['thresholds'][baseline.meta['selected_mode']]),
            'scope_coverage': coverage}, scores


def evaluate(output):
    if (output/'report.json').exists(): raise ValueError('Evaluation is frozen')
    p,meta,params = candidate(output)
    report = {'version': p['version'], 'scope': p['evaluation_scope'],
              'release_permission': False, 'automatic_archival': False,
              'protocol_sha256': sha(output/'protocol.json'), 'cohorts': {}, 'stop_reason': None}
    predictions = []
    if not meta['development_target_met']:
        report['stop_reason'] = 'development target failed; heldout features not read'
    else:
        for role in ('qualification', 'test', 'regression'):
            rows = [s for s in p['samples'] if s['role'] == role]
            result,scores = cohort(output, rows, meta, params)
            report['cohorts'][role] = result
            predictions.extend({'pixel_hash':s['pixel_hash'], 'role':role, 'truth':s['label'],
                                'score':float(score), 'prediction':'defect' if score>=meta['thresholds'][0] else 'normal'}
                               for s,score in zip(rows,scores))
            print({role:result}, flush=True)
            if previous.failed_gate(result['candidate'], result['scope_coverage'], p['quality_gate']):
                report['stop_reason'] = role+' quality gate failed; no further tuning'
                break
        if not report['stop_reason']:
            report['stop_reason'] = 'historical numerical gates passed; no new independent validation or production permission'
    report['historical_numerical_gates_passed'] = len(report['cohorts']) == 3 and all(
        not previous.failed_gate(m['candidate'],m['scope_coverage'],p['quality_gate']) for m in report['cohorts'].values())
    verify(output)
    write(output/'predictions.json', predictions); write(output/'report.json', report)
    print({'passed_historical_numeric_gates': report['historical_numerical_gates_passed'],
           'stop_reason': report['stop_reason'], 'production_permission': False}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','fit','evaluate'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(); {'prepare':prepare,'fit':fit,'evaluate':evaluate}[args.action](args.output.resolve())


if __name__ == '__main__': main()
