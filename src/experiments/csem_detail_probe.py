"""One normal-only detail probe on development specimens, never the used Test.

The previous source Test has already been evaluated. Results here are development
diagnostics only and cannot authorize publication, even when calibration improves.
"""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
from PIL import Image

from experiments.csem_screw_pilot import OUTPUT as BASE, verify as verify_base
from experiments.m2_body_quality import ROOT, save_arrays, measure
from perception.feature_backends import local_crops, representative_bank, cosine_anomalies, binary_probability
from products.families import read, write
from review.workflow import file_hash

OUTPUT = ROOT/'results/experiments/csem_detail_development_v1'
SOURCES = ('src/experiments/csem_detail_probe.py', 'src/perception/feature_backends.py',
           'src/perception/winclip_patch.py', 'weights/CLIP-ViT-B-16-open_clip_model.safetensors')


def development_rows(samples):
    """Filter before any image/cache access; a physical specimen has one role."""
    roles = {}
    for sample in samples:
        roles.setdefault(sample['specimen_id'], set()).add(sample['role'])
    if any(len(value) != 1 for value in roles.values()):
        raise ValueError('One physical specimen crossed development and evaluation roles')
    return [s for s in samples if s['role'] == 'calibration' or
            (s['role'] == 'fit' and s['label'] == 'normal')]


def prepare(output):
    p = verify_base(BASE)
    if not (BASE/'report.json').is_file():
        raise ValueError('This probe explicitly follows the prior completed source test')
    rows = development_rows(p['samples'])
    protocol = {'version': 'csem-detail-development-v1', 'samples': rows, 'seed': 20261107,
                'source_protocol_sha256': file_hash(BASE/'protocol.json'),
                'baseline_candidate_sha256': file_hash(BASE/'candidate.json'),
                'baseline_arrays_sha256': file_hash(BASE/'candidate.npz'),
                'baseline_development_sha256': file_hash(BASE/'development_report.json'),
                'frozen_sources': {name: file_hash(ROOT/name) for name in SOURCES},
                'method': 'four overlapping 60-percent crops; normal CLIP patch memory; max anomaly; no masks',
                'bank_size': 512, 'normal_calibration_quantile': .99, 'lights': ['501', '507', '901'],
                'local_bank_quality_labels': 'fit normal specimens only',
                'prior_test_seen': True, 'new_independent_evaluation': False,
                'automatic_archival': False, 'operational_release_ready': False,
                'stop_rule': 'one fixed detail method on existing development only; no parameter search or source Test reread'}
    if (output/'protocol.json').exists() and read(output/'protocol.json') != protocol:
        raise ValueError('Do not overwrite frozen detail probe')
    write(output/'protocol.json', protocol)
    print({'development_images': len(rows), 'roles': dict(Counter(s['role'] for s in rows)),
           'test_images': 0, 'candidate_count': 1}, flush=True)


def verify(output):
    p = read(output/'protocol.json')
    for relative, expected in p['frozen_sources'].items():
        if file_hash(ROOT/relative) != expected:
            raise ValueError('Frozen detail method changed')
    for name, key in [('protocol.json', 'source_protocol_sha256'), ('candidate.json', 'baseline_candidate_sha256'),
                      ('candidate.npz', 'baseline_arrays_sha256'), ('development_report.json', 'baseline_development_sha256')]:
        if file_hash(BASE/name) != p[key]:
            raise ValueError('Prior baseline changed')
    if development_rows(p['samples']) != p['samples'] or any(s['role'] not in ('fit', 'calibration') for s in p['samples']):
        raise ValueError('Probe contains non-development data')
    return p


def extract(output, rows):
    if any(s['role'] not in ('fit', 'calibration') or (s['role'] == 'fit' and s['label'] != 'normal') for s in rows):
        raise ValueError('Detail extraction may only read fit-normal and calibration images')
    cache = output/'features'; cache.mkdir(exist_ok=True)
    missing = [s for s in rows if not (cache/(s['pixel_hash']+'.npz')).is_file()]
    if not missing:
        return
    import torch
    from perception.winclip_patch import WinCLIPPatch
    torch.set_num_threads(2); detector = WinCLIPPatch(device='cpu'); started = time.monotonic()
    for index, s in enumerate(missing):
        path = ROOT/s['path']
        if file_hash(path) != s['file_sha256']:
            raise ValueError('Frozen development source changed')
        with Image.open(path) as original:
            tiles = local_crops(original.convert('RGB'))
        local, _ = detector.extract_batch(tiles)
        save_arrays(cache/(s['pixel_hash']+'.npz'), patches=local.reshape(-1, local.shape[-1]))
        if index % 12 == 0:
            print({'detail_images': index+1, 'total': len(missing), 'seconds': round(time.monotonic()-started)}, flush=True)


def patches(output, row):
    if row['role'] not in ('fit', 'calibration'):
        raise ValueError('Source Test cache is forbidden in this probe')
    with np.load(output/'features'/(row['pixel_hash']+'.npz'), allow_pickle=False) as data:
        return data['patches'].copy()


def run(output):
    if (output/'report.json').exists():
        raise ValueError('Preserve completed development probe')
    p = verify(output); rows = p['samples']; extract(output, rows)
    normal = [s for s in rows if s['role'] == 'fit']; cal = [s for s in rows if s['role'] == 'calibration']
    bank = representative_bank(np.concatenate([patches(output, s) for s in normal]), p['bank_size'], p['seed'])
    scores = np.array([float(cosine_anomalies(patches(output, s), bank).max()) for s in cal])
    y = np.array([s['label'] == 'defect' for s in cal], int)
    low = float(np.quantile(scores[y == 0], p['normal_calibration_quantile'])); high = low*1.5
    base = read(BASE/'candidate.json')
    with np.load(BASE/'candidate.npz', allow_pickle=False) as arrays:
        params = {key: arrays[key].copy() for key in arrays.files}
    baselines = {'normal_memory': [], 'global_binary': []}
    for s in cal:
        # Only explicitly selected calibration cache files, never a directory glob.
        with np.load(BASE/'features'/(s['pixel_hash']+'.npz'), allow_pickle=False) as data:
            baselines['normal_memory'].append(float(cosine_anomalies(data['patches'], params['bank']).max()))
            baselines['global_binary'].append(float(binary_probability(data['vector'], params, 'global')[0]))
    predictions = []
    for i, s in enumerate(cal):
        predictions.append({**s, 'local_score': float(scores[i]), 'local_prediction': 'defect' if scores[i] >= low else 'normal',
                            'baseline_scores': {mode: values[i] for mode, values in baselines.items()}})
    report = {}
    for light in p['lights']:
        indices = np.array([i for i, s in enumerate(cal) if s['illumination'] == light])
        report[light] = {'local_normal_memory': measure(y[indices], scores[indices], low, high),
                        **{mode: measure(y[indices], np.asarray(values)[indices], *base['thresholds'][mode])
                           for mode, values in baselines.items()}}
    recovered = [s for s in predictions if s['label'] == 'defect' and s['local_prediction'] == 'defect' and
                 s['baseline_scores']['global_binary'] < base['thresholds']['global_binary'][0]]
    save_arrays(output/'normal_detail_bank.npz', bank=bank)
    write(output/'calibration_predictions.json', predictions)
    result = {'version': p['version'], 'by_illumination': report, 'local_threshold': low, 'local_high_reference': high,
              'fit_normal_specimens': len({s['specimen_id'] for s in normal}),
              'calibration_specimens': len({s['specimen_id'] for s in cal}), 'calibration_paired_images': len(cal),
              'recovered_global_missed_calibration_views': [{'specimen_id': s['specimen_id'], 'illumination': s['illumination']} for s in recovered],
              'source_test_images_read': 0, 'prior_source_test_seen': True, 'new_independent_evaluation': False,
              'automatic_archival': False, 'operational_release_ready': False,
              'scope': 'development evidence only; not accuracy on new independent samples or release qualification',
              'protocol_sha256': file_hash(output/'protocol.json'), 'bank_sha256': file_hash(output/'normal_detail_bank.npz')}
    write(output/'report.json', result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run']); parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    {'prepare': prepare, 'run': run}[args.action](args.output)


if __name__ == '__main__':
    main()
