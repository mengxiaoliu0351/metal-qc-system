"""Two fixed CLIP baselines on one new screw view, split by physical specimen."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image

from experiments.m2_body_quality import ROOT, save_arrays, measure
from experiments.m2_body_robust import conservative_threshold
from perception.feature_backends import representative_bank, cosine_anomalies, binary_probability, export_binary
from products.families import read, write
from review.workflow import file_hash

OUTPUT = ROOT/'results/experiments/csem_screw_pilot_v1'
MANIFEST = ROOT/'data/external/csem_misd/pilot_manifest.json'
MODES = ('normal_memory', 'global_binary')


def specimen_roles(specimens, seed=20261107):
    roles = {s['specimen_id']: 'test' for s in specimens if s['publisher_partition'] == 'Test'}
    for label in ('normal', 'defect'):
        pool = sorted((s for s in specimens if s['publisher_partition'] == 'Train' and s['label'] == label),
                      key=lambda s: hashlib.sha256((str(seed)+s['specimen_id']).encode()).hexdigest())
        if len(pool) < 8: raise ValueError('Too few labelled Train specimens for the bounded pilot')
        n = int(np.ceil(len(pool)*.625)) if label == 'normal' else len(pool)//2
        for i, s in enumerate(pool): roles[s['specimen_id']] = 'fit' if i < n else 'calibration'
    return roles


def prepare(output):
    m = read(MANIFEST); roles = specimen_roles(m['specimens'])
    rows = [{**s, 'role': roles[s['specimen_id']]} for s in m['images']]
    if any({s['role'] for s in rows if s['specimen_id'] == key} != {role} for key, role in roles.items()):
        raise ValueError('Illumination images crossed specimen roles')
    if len({s['pixel_hash'] for s in rows}) != len(rows): raise ValueError('Exact duplicate images need audit before fitting')
    p = {'version': 'csem-screw-pilot-v1', 'family_key': 'screw', 'view': 'csem_slotted_head', 'samples': rows,
         'specimen_roles': roles, 'manifest_sha256': file_hash(MANIFEST), 'seed': 20261107,
         'primary_light': '501', 'robustness_lights': ['507', '901'], 'modes': list(MODES), 'bank_size': 512,
         'classifier_C': .1, 'calibration_target': {'minimum_recall': .85, 'maximum_false_positive_rate': .15},
         'research_gate': {'minimum_recall': .85, 'maximum_false_positive_rate': .15},
         'selection': 'calibration feasible first; balanced accuracy, recall, lower false-positive rate, normal-memory preference on ties',
         'preprocessing': 'original complete 512x512 RGB image, standard frozen CLIP preprocessing; no defect/foreground mask input',
         'weights_sha256': file_hash(ROOT/'weights/CLIP-ViT-B-16-open_clip_model.safetensors'),
         'extractor_sha256': file_hash(ROOT/'src/perception/winclip_patch.py'), 'implementation_sha256': file_hash(Path(__file__)),
         'physical_unit': 'one author sampleNr; three paired illuminations are not independent samples',
         'test_used_for_selection': False, 'automatic_archival': False, 'web_image_publication': False,
         'qualification': 'small independent source pilot only; no industrial release or automatic disposition qualification'}
    if (output/'protocol.json').exists() and read(output/'protocol.json') != p: raise ValueError('Do not overwrite frozen pilot protocol')
    write(output/'protocol.json', p)
    reserved_path = ROOT/'results/evaluation_reservations.json'; reserved = read(reserved_path)
    for fingerprint, entry in m['test_specimen_image_hashes'].items():
        reserved.setdefault('global', {}).setdefault(fingerprint, {
            'experiment': p['version'], 'role': 'test', 'dataset': m['dataset'], **entry,
            'scope': 'all illuminations of an officially held-out physical specimen, including non-pilot images'})
    write(reserved_path, reserved)
    print({'specimen_roles': dict(Counter(roles.values())), 'paired_test_images': sum(s['role'] == 'test' for s in rows)}, flush=True)


def verify(output):
    p = read(output/'protocol.json')
    for path, key in [(MANIFEST, 'manifest_sha256'), (Path(__file__), 'implementation_sha256'),
                      (ROOT/'src/perception/winclip_patch.py', 'extractor_sha256'),
                      (ROOT/'weights/CLIP-ViT-B-16-open_clip_model.safetensors', 'weights_sha256')]:
        if file_hash(path) != p[key]: raise ValueError('Frozen pilot input or implementation changed')
    return p


def extract_rows(output, rows):
    verify(output); cache = output/'features'; cache.mkdir(exist_ok=True)
    missing = [s for s in rows if not (cache/(s['pixel_hash']+'.npz')).exists()]
    if not missing: return
    import torch
    from perception.winclip_patch import WinCLIPPatch
    torch.set_num_threads(2); detector = WinCLIPPatch(device='cpu'); started = time.monotonic()
    for start in range(0, len(missing), 4):
        batch = missing[start:start+4]; images = []
        for s in batch:
            if file_hash(ROOT/s['path']) != s['file_sha256']: raise ValueError('Frozen source image changed')
            with Image.open(ROOT/s['path']) as image: images.append(image.convert('RGB'))
        patches, vectors = detector.extract_batch(images)
        for s, local, vector in zip(batch, patches, vectors):
            save_arrays(cache/(s['pixel_hash']+'.npz'), patches=local, vector=vector)
        if start % 24 == 0: print({'features': start+len(batch), 'total': len(missing), 'seconds': round(time.monotonic()-started)}, flush=True)


def features(output, rows, bank):
    x = []; raw = []
    for s in rows:
        with np.load(output/'features'/(s['pixel_hash']+'.npz'), allow_pickle=False) as z:
            x.append(z['vector']); raw.append(float(cosine_anomalies(z['patches'], bank).max()))
    return np.asarray(x), np.asarray(raw)


def fit(output):
    if (output/'candidate.json').exists(): raise ValueError('Pilot already fitted; no refitting after test')
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    p = verify(output); train = [s for s in p['samples'] if s['role'] == 'fit']; cal = [s for s in p['samples'] if s['role'] == 'calibration']
    extract_rows(output, train+cal)
    parts = []
    for s in train:
        if s['label'] == 'normal':
            with np.load(output/'features'/(s['pixel_hash']+'.npz'), allow_pickle=False) as z: parts.append(z['patches'])
    bank = representative_bank(np.concatenate(parts), p['bank_size'], p['seed'])
    x, raw = features(output, train, bank); xc, rc = features(output, cal, bank)
    y = np.array([s['label'] == 'defect' for s in train], int); yc = np.array([s['label'] == 'defect' for s in cal], int)
    pipeline = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(C=p['classifier_C'], class_weight='balanced', max_iter=2000, random_state=p['seed']))])
    weights = np.array([1/sum(r['specimen_id'] == s['specimen_id'] for r in train) for s in train])
    with threadpool_limits(limits=2): pipeline.fit(x, y, clf__sample_weight=weights)
    params = export_binary(pipeline, 'global'); params['bank'] = bank
    binary = binary_probability(xc, params, 'global'); low, high, feasible = conservative_threshold(yc, binary, p['calibration_target'])
    normal_low = float(np.quantile(rc[yc == 0], .99)); normal_high = max(normal_low*1.5, float(np.nextafter(normal_low, np.inf)))
    thresholds = {'normal_memory': [normal_low, normal_high], 'global_binary': [low, high]}
    metrics = {mode: measure(yc, rc if mode == 'normal_memory' else binary, *thresholds[mode]) for mode in MODES}
    choices = []
    for i, mode in enumerate(MODES):
        m = metrics[mode]; good = m['recall'] >= .85 and m['false_positive_rate'] <= .15
        choices.append(((int(good), m['balanced_accuracy'], m['recall'], -m['false_positive_rate'], -i), mode))
    selected = max(choices)[1]; save_arrays(output/'candidate.npz', **params)
    write(output/'candidate.json', {'version': p['version'], 'selected': selected, 'thresholds': thresholds,
          'arrays_sha256': file_hash(output/'candidate.npz'), 'protocol_sha256': file_hash(output/'protocol.json'),
          'automatic_archival': False, 'requires_review': True})
    write(output/'development_report.json', {'calibration': metrics, 'selected': selected, 'test_features_read': False,
          'training_specimens': len({s['specimen_id'] for s in train}), 'calibration_specimens': len({s['specimen_id'] for s in cal}),
          'label_budget_note': 'normal-memory bank uses only fitted normal specimens; binary arm additionally uses source defect labels; shared frozen CLIP'})
    print({'selected': selected, 'calibration': metrics, 'test_features_read': False}, flush=True)


def evaluate(output):
    if (output/'report.json').exists(): raise ValueError('Pilot test report is frozen')
    p = verify(output); model = read(output/'candidate.json')
    if file_hash(output/'candidate.npz') != model['arrays_sha256'] or file_hash(output/'protocol.json') != model['protocol_sha256']:
        raise ValueError('Candidate artifact changed')
    with np.load(output/'candidate.npz', allow_pickle=False) as z: params = {k: z[k].copy() for k in z.files}
    rows = [s for s in p['samples'] if s['role'] == 'test']; extract_rows(output, rows)
    report = {}; predictions = []
    for light in (p['primary_light'], *p['robustness_lights']):
        part = [s for s in rows if s['illumination'] == light]; x, raw = features(output, part, params['bank'])
        y = np.array([s['label'] == 'defect' for s in part], int)
        scores = {'normal_memory': raw, 'global_binary': binary_probability(x, params, 'global')}
        report[light] = {mode: measure(y, scores[mode], *model['thresholds'][mode]) for mode in MODES}
        for i, s in enumerate(part): predictions.append({**s, 'scores': {mode: float(scores[mode][i]) for mode in MODES}})
    selected = model['selected']; eligible = all(m[selected]['recall'] >= .85 and m[selected]['false_positive_rate'] <= .15 for m in report.values())
    result = {'version': p['version'], 'selected': selected, 'primary_light': p['primary_light'], 'by_illumination': report,
              'independent_test_specimens': len({s['specimen_id'] for s in rows}), 'paired_test_images': len(rows),
              'eligible_for_research_hint': eligible, 'automatic_archival': False, 'view_routing_validated': False,
              'operational_release_ready': False, 'web_image_publication': False,
              'scope': 'one new CSEM screw-head view; official specimen-level Test; paired lights are repeated views, not extra independent units; no industrial or all-model claim',
              'development': read(output/'development_report.json'), 'protocol_sha256': model['protocol_sha256'], 'arrays_sha256': model['arrays_sha256']}
    write(output/'test_predictions.json', predictions); write(output/'report.json', result)
    print({'selected': selected, 'by_illumination': report, 'research_gate': eligible, 'operational_release_ready': False}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'fit', 'evaluate']); parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    {'prepare': prepare, 'fit': fit, 'evaluate': evaluate}[args.action](args.output)


if __name__ == '__main__': main()
