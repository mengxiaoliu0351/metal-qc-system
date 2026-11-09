"""Frozen detail-feature development comparison. ALL evaluation cohorts are historical.

No production pointer, quality permission, reservation, or source label is changed.
Passing historical gates is insufficient to publish without new independent data.
"""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from experiments.m2_body_quality import ROOT, save_arrays, load_features, measure
from experiments.m2_body_robust import conservative_threshold
from experiments.prepare_m2_body import sha
from perception.body_detail import detail_features, prepare_detail
from perception.body_preprocess import prepare_body
from perception.body_quality import BodyQualityModel, body_scores
from perception.feature_backends import binary_probability, export_binary
from products.families import read, write

PRIOR = ROOT/'results/experiments/m2_body_quality_v2'
OUTPUT = ROOT/'results/experiments/m2_body_quality_v3_detail'
MODES = ('detail_hog', 'detail_hog_clip')
CS = (.01, .1, 1.)


def prepare(output):
    old = read(PRIOR/'protocol.json')
    reserved = read(ROOT/'results/evaluation_reservations.json').get('global', {})
    samples = [s for s in old['samples'] if s['view'] == 'body']
    if any(s['pixel_hash'] in reserved for s in samples if s['role'] in ('train', 'calibration')):
        raise ValueError('Globally reserved image cannot enter training or calibration')
    if len({s['pixel_hash'] for s in samples}) != len(samples):
        raise ValueError('Image crossed experiment roles')
    p = {'version': 'm2-body-quality-v3-detail-development', 'samples': samples,
         'prior_protocol_sha256': sha(PRIOR/'protocol.json'), 'prior_model_sha256': sha(PRIOR/'quality.npz'),
         'preprocessing_sha256': sha(ROOT/'src/perception/body_detail.py'),
         'implementation_sha256': sha(Path(__file__)), 'seed': 20261107,
         'modes': list(MODES), 'classifier_C': list(CS),
         'calibration_target': {'minimum_recall': .93, 'maximum_false_positive_rate': .08},
         'historical_gate': {'minimum_recall': .9, 'maximum_false_positive_rate': .1, 'minimum_scope_coverage': .95},
         'selection': 'same calibration-only conservative objective as V2; six fixed candidates; prefer detail-only and stronger regularization on ties',
         'evaluation_status': 'all three cohorts were previously evaluated in V1/V2; historical regression only, no new blind test',
         'independent_new_test_available': False, 'eligible_for_production': False,
         'automatic_archival': False, 'source_label_basis': 'publisher OK/NG',
         'supervision': 'labelled lightweight classifier, not a normal-only or zero-shot method',
         'physical_identity': 'not provided; image-level separation only'}
    path = output/'protocol.json'
    if path.exists() and read(path) != p:
        raise ValueError('Do not overwrite frozen protocol')
    write(path, p)
    print({'roles': dict(Counter(s['role'] for s in samples)), 'new_blind_test': False}, flush=True)


def verify(output):
    p = read(output/'protocol.json')
    if sha(ROOT/'src/perception/body_detail.py') != p['preprocessing_sha256'] or sha(Path(__file__)) != p['implementation_sha256']:
        raise ValueError('Frozen implementation changed')
    if sha(PRIOR/'protocol.json') != p['prior_protocol_sha256'] or sha(PRIOR/'quality.npz') != p['prior_model_sha256']:
        raise ValueError('Frozen baseline changed')
    return p


def extract_rows(output, rows):
    verify(output)
    cache = output/'features'; cache.mkdir(exist_ok=True)
    start = time.monotonic()
    for i, sample in enumerate(rows):
        path = cache/(sample['pixel_hash']+'.npz')
        if path.exists():
            continue
        if sha(ROOT/sample['path']) != sample['file_sha256']:
            raise ValueError('Frozen source image changed')
        with Image.open(ROOT/sample['path']) as image:
            vector, metadata = detail_features(image)
        # A failed proposal remains a failed/zero representation; never remove
        # difficult images from binary evaluation or count rejection as correct.
        if vector is None:
            vector = np.zeros(31*11*36, dtype=np.float32)
        save_arrays(path, detail=vector, proposal_found=np.array(metadata['proposal_found']),
                    source_box=np.array(metadata['source_box'] or [0, 0, 0, 0]))
        if (i+1) % 100 == 0:
            print({'extracted': i+1, 'total': len(rows), 'seconds': round(time.monotonic()-start)}, flush=True)


def extract(output):
    p = verify(output)
    extract_rows(output, [s for s in p['samples'] if s['role'] in ('train', 'calibration')])


def features(output, rows, mode):
    values = []; found = []
    for s in rows:
        with np.load(output/'features'/(s['pixel_hash']+'.npz'), allow_pickle=False) as z:
            values.append(z['detail']); found.append(bool(z['proposal_found']))
    x = np.asarray(values)
    if mode == 'detail_hog_clip':
        clip, _ = load_features(PRIOR, rows)
        x = np.c_[x, clip]
    elif mode != 'detail_hog':
        raise ValueError('Unknown feature mode')
    return x, np.asarray(found)


def fit(output):
    if (output/'model.json').exists():
        raise ValueError('Do not refit frozen detail experiment')
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    p = verify(output); train = [s for s in p['samples'] if s['role'] == 'train']
    cal = [s for s in p['samples'] if s['role'] == 'calibration']
    y = np.array([s['label'] == 'defect' for s in train], int)
    yc = np.array([s['label'] == 'defect' for s in cal], int)
    candidates = {}; arrays = {}; choices = []
    for mode in p['modes']:
        x, _ = features(output, train, mode); xc, _ = features(output, cal, mode)
        for c in p['classifier_C']:
            name = f'{mode}_c{c:g}'.replace('.', 'p')
            pipeline = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(C=c, class_weight='balanced',
                                       max_iter=3000, random_state=p['seed'], solver='liblinear'))])
            with threadpool_limits(limits=2):
                pipeline.fit(x, y)
            params = export_binary(pipeline, name); arrays.update(params)
            score = binary_probability(xc, params, name)
            low, high, feasible = conservative_threshold(yc, score, p['calibration_target'])
            m = measure(yc, score, low, high)
            candidates[name] = {'mode': mode, 'C': c, 'thresholds': [low, high], 'target_feasible': feasible, 'calibration': m}
            rank = (int(feasible), -m['false_positive_rate'], m['recall'], -c, int(mode == 'detail_hog')) if feasible else (0, m['balanced_accuracy'], m['recall'], -c, int(mode == 'detail_hog'))
            choices.append((rank, name))
    selected = max(choices)[1]; chosen = candidates[selected]
    save_arrays(output/'candidates.npz', **arrays)
    model_arrays = {k.replace(selected+'_', 'detail_', 1): v for k, v in arrays.items() if k.startswith(selected+'_')}
    save_arrays(output/'model.npz', **model_arrays)
    metadata = {'version': p['version'], 'selected': selected, **chosen, 'arrays_sha256': sha(output/'model.npz'),
                'protocol_sha256': sha(output/'protocol.json'), 'automatic_archival': False, 'eligible_for_production': False}
    write(output/'model.json', metadata)
    write(output/'development_report.json', {'selected': selected, 'candidates': candidates,
          'training_images': len(train), 'calibration_images': len(cal), 'evaluation_features_read': False,
          'new_blind_test': False, 'automatic_archival': False})
    print({'selected': selected, 'calibration': chosen['calibration'], 'evaluation_features_read': False}, flush=True)


def evaluate(output):
    if (output/'report.json').exists():
        raise ValueError('Historical evaluation is frozen')
    p = verify(output); meta = read(output/'model.json')
    if sha(output/'model.npz') != meta['arrays_sha256'] or sha(output/'protocol.json') != meta['protocol_sha256']:
        raise ValueError('Detail artifact integrity failure')
    with np.load(output/'model.npz', allow_pickle=False) as z:
        params = {k: z[k].copy() for k in z.files}
    prior = BodyQualityModel(PRIOR/'quality.json'); reports = {}; predictions = []
    for role in ('qualification', 'test', 'regression'):
        rows = [s for s in p['samples'] if s['role'] == role]; extract_rows(output, rows)
        x, found = features(output, rows, meta['mode']); y = np.array([s['label'] == 'defect' for s in rows], int)
        scores = binary_probability(x, params, 'detail'); low, high = meta['thresholds']
        m = measure(y, scores, low, high)
        xp, rp = load_features(PRIOR, rows); old_score = body_scores(xp, rp, prior.params, prior.meta['selected_mode'])
        old = measure(y, old_score, *prior.meta['thresholds'][prior.meta['selected_mode']])
        gate = p['historical_gate']; passed = m['recall'] >= gate['minimum_recall'] and m['false_positive_rate'] <= gate['maximum_false_positive_rate'] and float(found.mean()) >= gate['minimum_scope_coverage']
        reports[role] = {'n': len(rows), 'baseline_v2': old, 'detail_candidate': m,
                         'proposal_coverage': float(found.mean()), 'historical_binary_gate_passed': bool(passed),
                         'new_blind_test': False, 'scope': 'historical BODY regression; proposal coverage is not view rejection or localization accuracy'}
        for s, score, old_value, ok in zip(rows, scores, old_score, found):
            predictions.append({'pixel_hash': s['pixel_hash'], 'path': s['path'], 'role': role, 'truth': s['label'],
                                'score': float(score), 'baseline_score': float(old_value), 'proposal_found': bool(ok),
                                'prediction': 'defect' if score >= low else 'normal'})
    passed = all(r['historical_binary_gate_passed'] for r in reports.values())
    report = {'version': p['version'], 'selected': meta['selected'], 'cohorts': reports,
              'all_historical_binary_gates_passed': passed, 'independent_new_test_available': False,
              'eligible_for_production': False, 'automatic_archival': False,
              'release_blockers': (['historical binary gate failed'] if not passed else []) +
                 ['no new independent evaluation data', 'new representation has not been integrated with view gating or validated for automatic disposition'],
              'arrays_sha256': meta['arrays_sha256'], 'protocol_sha256': meta['protocol_sha256'],
              'scope': p['evaluation_status'], 'labels_modified': False, 'production_modified': False}
    write(output/'historical_predictions.json', predictions); write(output/'report.json', report)
    print({role: {'accuracy': r['detail_candidate']['accuracy'], 'recall': r['detail_candidate']['recall'],
                  'false_positive_rate': r['detail_candidate']['false_positive_rate']} for role, r in reports.items()}, flush=True)
    print({'historical_gates_passed': passed, 'eligible_for_production': False}, flush=True)


def diagnose(output):
    """Deterministic non-cherry-picked historical error examples and crop audit."""
    p = verify(output); prior = BodyQualityModel(PRIOR/'quality.json')
    rows = [s for s in p['samples'] if s['role'] in ('calibration', 'qualification', 'test', 'regression')]
    groups = {}; strata = {}; boxes = []
    for role in ('calibration', 'qualification', 'test', 'regression'):
        part = [s for s in rows if s['role'] == role]; x, raw = load_features(PRIOR, part)
        score = body_scores(x, raw, prior.params, prior.meta['selected_mode']); low = prior.meta['thresholds']['global_binary'][0]
        for sample, value in zip(part, score):
            with Image.open(ROOT/sample['path']) as im:
                _, old = prepare_body(im); _, detail = prepare_detail(im)
            ob = old['source_box']; nb = detail['source_box']
            ratio = (ob[3]-ob[1])/(nb[3]-nb[1]) if nb else None
            item = {'pixel_hash': sample['pixel_hash'], 'role': role, 'truth': sample['label'], 'score': float(value),
                    'old_box': ob, 'detail_box': nb, 'old_to_detail_height': ratio}
            boxes.append(item)
            error = ('FN' if value < low else 'TP') if sample['label'] == 'defect' else ('TN' if value < low else 'FP')
            key = role+':'+error
            strata.setdefault(key, []).append(ratio)
            if role == 'regression': groups.setdefault(error, []).append((sample, float(value)))
    summary = {key: {'n': len(values), 'detail_proposal_failed': sum(v is None for v in values),
                     'old_height_less_than_70pct_detail': sum(v is not None and v < .7 for v in values),
                     'median_height_ratio': float(np.median([v for v in values if v is not None])) if any(v is not None for v in values) else None}
               for key, values in strata.items()}
    canvas = Image.new('RGB', (1440, 1260), 'white'); draw = ImageDraw.Draw(canvas)
    for gi, group in enumerate(('FP', 'TN', 'FN', 'TP')):
        for j, (sample, value) in enumerate(sorted(groups[group], key=lambda v: v[0]['pixel_hash'])[:6]):
            with Image.open(ROOT/sample['path']) as im:
                old, _ = prepare_body(im); detail, _ = prepare_detail(im)
                thumb = ImageOps.contain(im.convert('RGB'), (75, 250))
                canvas.paste(thumb, (j*240, gi*310+35))
                canvas.paste(old.resize((80, 250)), (j*240+78, gi*310+35))
                if detail is not None: canvas.paste(detail.resize((80, 250)).convert('RGB'), (j*240+160, gi*310+35))
            draw.text((j*240+3, gi*310+3), f'{group} {Path(sample["path"]).stem} {value:.3f}', fill='black')
            draw.text((j*240+3, gi*310+290), 'source / V2 / detail', fill='black')
    canvas.save(output/'crop_audit.png')
    write(output/'crop_diagnosis.json', {'strata': summary, 'rows': boxes,
          'causal_claim': False, 'scope': 'proposal differences and prediction associations; no box truth or causal attribution'})
    print({'crop_audit': str(output/'crop_audit.png'), 'strata': summary}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'diagnose', 'extract', 'fit', 'evaluate'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    {'prepare': prepare, 'diagnose': diagnose, 'extract': extract, 'fit': fit, 'evaluate': evaluate}[args.action](args.output)


if __name__ == '__main__':
    main()
