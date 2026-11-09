"""Calibration-only fixed score mixtures; historical development, never a release.

This follow-up was proposed after seeing V3 fail. No cohort here is a new blind
test, and passing a historical numerical gate does not confer quality permission.
"""
import argparse
from pathlib import Path

import numpy as np

from experiments.m2_body_quality import ROOT, load_features, measure
from experiments.m2_body_robust import conservative_threshold
from experiments import m2_body_detail as detail
from experiments.prepare_m2_body import sha
from perception.body_quality import BodyQualityModel, body_scores
from perception.feature_backends import binary_probability
from products.families import read, write

OUTPUT = ROOT/'results/experiments/m2_body_quality_v4_score_fusion'
WEIGHTS = (0., .1, .25, .5, .75, 1.)


def dependencies():
    return {str(path.relative_to(ROOT)): sha(path) for path in
            [detail.PRIOR/'protocol.json', detail.PRIOR/'quality.json', detail.PRIOR/'quality.npz',
             detail.OUTPUT/'protocol.json', detail.OUTPUT/'model.json', detail.OUTPUT/'model.npz',
             ROOT/'src/experiments/m2_body_fusion.py']}


def prepare(output):
    p = {'version': 'm2-body-v4-score-fusion-development', 'dependencies': dependencies(),
         'detail_weights': list(WEIGHTS),
         'target': {'minimum_recall': .93, 'maximum_false_positive_rate': .08},
         'selection': 'calibration only; feasible target first, then lower false positive rate, higher recall, smaller detail weight',
         'proposed_after': 'V3 historical regression failure; all evaluations remain development regressions',
         'new_blind_test': False, 'automatic_archival': False, 'eligible_for_production': False}
    if (output/'protocol.json').exists() and read(output/'protocol.json') != p:
        raise ValueError('Do not overwrite frozen fusion protocol')
    write(output/'protocol.json', p)


def verify(output):
    p = read(output/'protocol.json')
    if p['dependencies'] != dependencies():
        raise ValueError('Frozen fusion dependency changed')
    return p


def inputs(role):
    p = read(detail.OUTPUT/'protocol.json'); meta = read(detail.OUTPUT/'model.json')
    rows = [s for s in p['samples'] if s['role'] == role]
    prior = BodyQualityModel(detail.PRIOR/'quality.json')
    x, raw = load_features(detail.PRIOR, rows)
    base = body_scores(x, raw, prior.params, prior.meta['selected_mode'])
    xd, found = detail.features(detail.OUTPUT, rows, meta['mode'])
    with np.load(detail.OUTPUT/'model.npz', allow_pickle=False) as z:
        params = {k: z[k].copy() for k in z.files}
    scores = binary_probability(xd, params, 'detail')
    return rows, np.array([s['label'] == 'defect' for s in rows], int), base, scores, found


def fit(output):
    if (output/'model.json').exists():
        raise ValueError('Do not refit frozen fusion')
    p = verify(output); _, y, base, scores, _ = inputs('calibration')
    candidates = []; choices = []
    for weight in p['detail_weights']:
        mixed = (1-weight)*base+weight*scores
        low, high, feasible = conservative_threshold(y, mixed, p['target'])
        m = measure(y, mixed, low, high)
        row = {'detail_weight': weight, 'thresholds': [low, high], 'target_feasible': feasible, 'calibration': m}
        rank = (int(feasible), -m['false_positive_rate'], m['recall'], -weight) if feasible else (0, m['balanced_accuracy'], m['recall'], -weight)
        candidates.append(row); choices.append((rank, row))
    chosen = max(choices, key=lambda item: item[0])[1]
    write(output/'model.json', {'version': p['version'], **chosen, 'protocol_sha256': sha(output/'protocol.json'),
                               'automatic_archival': False, 'eligible_for_production': False})
    write(output/'development_report.json', {'candidates': candidates, 'selected': chosen, 'historical_cohorts_read': False})
    print({'selected': chosen, 'historical_cohorts_read': False}, flush=True)


def evaluate(output):
    if (output/'report.json').exists():
        raise ValueError('Historical fusion evaluation is frozen')
    p = verify(output); model = read(output/'model.json')
    if model['protocol_sha256'] != sha(output/'protocol.json'):
        raise ValueError('Fusion protocol changed')
    weight = model['detail_weight']; results = {}; predictions = []
    for role in ('qualification', 'test', 'regression'):
        rows, y, base, scores, found = inputs(role)
        mixed = (1-weight)*base+weight*scores
        m = measure(y, mixed, *model['thresholds'])
        passed = m['recall'] >= .9 and m['false_positive_rate'] <= .1 and (weight == 0 or float(found.mean()) >= .95)
        matrix = np.asarray(m['confusion_matrix'])
        results[role] = {'n': len(rows), 'metrics': m, 'historical_binary_gate_passed': bool(passed),
                         'detail_proposal_coverage': float(found.mean()),
                         'detail_missing': {label: int(np.sum((y == value) & ~found)) for label, value in [('normal', 0), ('defect', 1)]},
                         'baseline_v2': read(detail.OUTPUT/'report.json')['cohorts'][role]['baseline_v2'],
                         'detail_v3': read(detail.OUTPUT/'report.json')['cohorts'][role]['detail_candidate'],
                         'new_blind_test': False}
        for s, score, ok in zip(rows, mixed, found):
            predictions.append({'pixel_hash': s['pixel_hash'], 'role': role, 'truth': s['label'], 'score': float(score),
                                'detail_proposal_found': bool(ok), 'prediction': 'defect' if score >= model['thresholds'][0] else 'normal'})
    passed = all(r['historical_binary_gate_passed'] for r in results.values())
    report = {'version': p['version'], 'selected': model, 'cohorts': results,
              'all_historical_binary_gates_passed': passed, 'new_blind_test': False,
              'eligible_for_production': False, 'automatic_archival': False,
              'release_blockers': ([] if passed else ['historical binary or proposal coverage gate failed']) +
                  ['new independent evaluation data unavailable', 'view gating and automatic disposition not validated'],
              'protocol_sha256': sha(output/'protocol.json'), 'model_sha256': sha(output/'model.json'),
              'production_modified': False, 'labels_modified': False}
    write(output/'historical_predictions.json', predictions); write(output/'report.json', report)
    print({role: value['metrics'] for role, value in results.items()}, flush=True)
    print({'historical_gates_passed': passed, 'eligible_for_production': False}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'fit', 'evaluate'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    {'prepare': prepare, 'fit': fit, 'evaluate': evaluate}[args.action](args.output)


if __name__ == '__main__':
    main()
