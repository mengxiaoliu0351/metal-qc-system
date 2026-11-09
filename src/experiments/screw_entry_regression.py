"""Frozen historical regression through the real family entry in isolated storage."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from products.families import read, write
from review.workflow import file_hash, image_hash

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT/'results'
OUTPUT = LIVE/'experiments/screw_entry_regression_v1'


def prepare(output):
    if (output/'protocol.json').exists():
        print('Existing frozen protocol retained', flush=True); return
    rows = []
    def add(path, cohort, truth, view, unit=None):
        path = Path(path).resolve()
        rows.append({'path': path.relative_to(ROOT).as_posix(), 'cohort': cohort, 'truth': truth,
                     'expected_view': view, 'physical_unit': unit,
                     'pixel_hash': image_hash(path), 'file_sha256': file_hash(path)})
    split = read(LIVE/'experiments/perception_upgrade_v1/screw_split.json')
    for relative in split['test']:
        path = ROOT/relative
        add(path, 'mvtec_screw', 'normal' if path.parent.name == 'good' else 'defect', 'mvtec_screw')
    for name in ('head', 'body'):
        for s in read(ROOT/f'data/external/m2_v2/{name}-manifest.json')['images']:
            if s['source_partition'] != 'Test': continue
            cohort = 'm2_'+name+('_preview' if s['role'] == 'layout_preview' else '')
            add(ROOT/s['source_path'], cohort, s['label'], 'm2_'+name)
    for s in read(ROOT/'data/external/csem_misd/pilot_manifest.json')['images']:
        if s['publisher_partition'] == 'Test':
            add(ROOT/s['path'], 'csem_uncovered', s['label'], None, s['specimen_id'])
    for s in read(LIVE/'product_families/screw/appearance.json')['samples']:
        if s['split'] == 'test':
            add(LIVE/s['path'], 'appearance_no_quality_truth', None, None, s['source_frame'])
    if len(rows) != 2163 or len({s['pixel_hash'] for s in rows}) != len(rows):
        raise ValueError('Regression count or identity audit changed; do not silently alter cohort')
    assets = set()
    for product in read(LIVE/'perception/banks/products.json'):
        assets.add(LIVE/'perception/banks'/product['bank'])
        meta = LIVE/'perception/optimized'/product['perception_artifact']
        assets.update((meta, meta.parent/read(meta)['arrays']))
        if product.get('review_benchmark'):
            assets.update(p for p in (LIVE/product['review_benchmark']).iterdir() if p.suffix in ('.json', '.npz'))
    assets.update(LIVE/'experiments/perception_upgrade_v1'/f'{p}_{suffix}' for p in ('metal_nut', 'screw', 'grid')
                  for suffix in ('split.json', 'features.npz'))
    for pointer, names in [('operational_router.json', ('router.json', 'router.npz', 'protocol.json', 'report.json')),
                           ('appearance_active.json', ('appearance.json', 'appearance.npz', 'report.json'))]:
        path = LIVE/'product_families/screw'/pointer; assets.add(path)
        assets.update(LIVE/read(path)['directory']/name for name in names)
    assets.update(LIVE/name for name in ('perception/banks/products.json', 'inspection_policy.json',
                                         'evaluation_reservations.json', 'data_intake/known_image_derivatives.json'))
    sources = ['src/web/family_inspection.py', 'src/web/server.py', 'src/review/disposition.py',
               'src/perception/winclip_patch.py', 'src/perception/feature_backends.py',
               'src/cognition/vlm_api.py', 'src/cognition/visual_input.py', 'src/cognition/knowledge_sources.py',
               'src/perception/appearance.py', 'src/perception/family_router.py', 'src/review/calibration_exclusions.json',
               'src/experiments/screw_entry_regression.py']
    protocol = {'version': 'screw-entry-historical-regression-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'samples': rows, 'cohort_counts': dict(Counter(s['cohort'] for s in rows)),
                'assets': {str(p.relative_to(LIVE).as_posix()): file_hash(p) for p in sorted(assets)},
                'sources': {name: file_hash(ROOT/name) for name in sources},
                'weights_sha256': file_hash(ROOT/'weights/CLIP-ViT-B-16-open_clip_model.safetensors'),
                'record_scope': 'evaluation', 'maximum_vlm_calls': 200,
                'scope': 'historical image-level regression of existing website backend entry; no fitting, no new blind or factory test',
                'missing_quality_truth': 'appearance crops only assess whether an unsupported quality model was applied',
                'paired_lights': 'CSEM 114 images are 38 physical specimens; they are not 114 independent units',
                'abstention_rule': 'abstention is neither correct nor a false negative; report it separately and in whole-cohort coverage',
                'live_models_or_policy_modified': False}
    output.mkdir(parents=True, exist_ok=True); write(output/'protocol.json', protocol)
    print({'frozen_images': len(rows), 'cohorts': protocol['cohort_counts']}, flush=True)


def verify(output):
    p = read(output/'protocol.json')
    for name, expected in p['sources'].items():
        if file_hash(ROOT/name) != expected: raise ValueError('Frozen source changed: '+name)
    for name, expected in p['assets'].items():
        if file_hash(LIVE/name) != expected: raise ValueError('Live model or policy changed: '+name)
    return p


def summarize_rows(rows):
    groups = {}
    for cohort in sorted({r['sample']['cohort'] for r in rows}):
        part = [r for r in rows if r['sample']['cohort'] == cohort]
        completed = [r for r in part if r['status'] == 'completed']
        classified = [r for r in completed if r['valid_quality_prediction']]
        labelled = [r for r in completed if r['sample']['truth'] in ('normal', 'defect')]
        tn=fp=fn=tp=0
        for r in classified:
            truth = r['sample']['truth']; predicted = r['result']['perception_prediction']
            tn += truth == 'normal' and predicted == 'normal'; fp += truth == 'normal' and predicted == 'defect'
            fn += truth == 'defect' and predicted == 'normal'; tp += truth == 'defect' and predicted == 'defect'
        times = [r['wall_seconds'] for r in completed]
        g = {'processed': len(part), 'completed': len(completed), 'errors': len(part)-len(completed),
             'quality_labelled': len(labelled), 'valid_classified': len(classified),
             'quality_abstained': len(completed)-len(classified),
             'misapplied_quality_models': sum(r.get('misapplied_quality_model', False) for r in completed),
             'confusion_matrix_classified_only': [[tn,fp],[fn,tp]],
             'accuracy_classified_only': (tn+tp)/len(classified) if classified else None,
             'correct_quality_decisions_over_labelled_all': (tn+tp)/len(labelled) if labelled else None,
             'false_positives': fp, 'false_negatives': fn,
             'defects_abstained': sum(r['sample']['truth']=='defect' and not r['valid_quality_prediction'] for r in completed),
             'vlm_triggered': sum(bool(r['result'].get('vlm_called')) for r in completed),
             'vlm_successful': sum(bool(r['result'].get('vlm_called') and r['result'].get('cognition',{}).get('ok')) for r in completed),
             'sent_to_review': sum(bool(r['result'].get('disposition',{}).get('requires_review')) for r in completed),
             'automatic_proposals': sum(r['result'].get('disposition',{}).get('auto_result') in ('normal','defect') for r in completed),
             'automatic_wrong': sum(r['sample']['truth'] in ('normal','defect') and
                                    r['result'].get('disposition',{}).get('auto_result') in ('normal','defect') and
                                    r['result']['disposition']['auto_result'] != r['sample']['truth'] for r in completed),
             'routing': dict(Counter(r['result'].get('family_routing',{}).get('view_key') or 'unmatched' for r in completed)),
             'review_reasons': dict(Counter(r['result'].get('disposition',{}).get('reason_code','unknown') for r in completed)),
             'median_backend_wall_seconds': float(np.median(times)) if times else None,
             'p95_backend_wall_seconds': float(np.quantile(times,.95)) if times else None}
        groups[cohort] = g
    return groups


def summarize(output):
    p = read(output/'protocol.json'); rows = [read(path) for path in sorted((output/'cases').glob('*.json'))]
    result = {'version': p['version'], 'expected': len(p['samples']), 'processed': len(rows),
              'complete': len(rows) == len(p['samples']), 'cohorts': summarize_rows(rows),
              'protocol_sha256': file_hash(output/'protocol.json'), 'scope': p['scope'],
              'timing_scope': 'local backend, sequential inference and record persistence, including actual VLM round trips; not browser/cloud or operator time',
              'human_verdicts_submitted': False, 'models_fitted_or_published': False}
    write(output/('report.json' if result['complete'] else 'partial_report.json'), result)
    return result


def run(output):
    p = verify(output); runtime = output/'runtime'; runtime.mkdir(exist_ok=True)
    for relative, expected in p['assets'].items():
        target = runtime/relative; target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): shutil.copy2(LIVE/relative, target)
        if file_hash(target) != expected: raise ValueError('Isolated asset changed: '+relative)
    import torch
    from web import server
    from web.family_inspection import inspect
    from interface import report
    from review.batches import BATCH_CONTEXT
    import config
    if not server.vlm_api.DASHSCOPE_API_KEY or 'PASTE' in server.vlm_api.DASHSCOPE_API_KEY:
        raise ValueError('Real VLM key not configured; do not fabricate full-system results')
    torch.set_num_threads(2)
    server.RESULTS_DIR = str(runtime); server.BANKS_DIR = str(runtime/'perception/banks')
    report.HISTORY_DIR = str(runtime/'history')
    server._products = None; server._review_workflow = None
    # This isolated process uses the same inference functions and model files.
    # No predictor, threshold, retrieval or VLM response is replaced.
    original = server.vlm_api.analyze; calls = [0]; failures = [0]
    prior = [read(path) for path in (output/'cases').glob('*.json')]
    calls[0] = sum(r.get('result',{}).get('vlm_called',False) for r in prior)
    def counted(*args, **kwargs):
        if calls[0] >= p['maximum_vlm_calls']: raise RuntimeError('Frozen VLM budget reached; stop regression')
        calls[0] += 1; response = original(*args, **kwargs)
        failures[0] = 0 if response.get('ok') else failures[0]+1
        return response
    server.vlm_api.analyze = counted
    write(output/'environment.json', {'torch': torch.__version__, 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
          'torch_threads': 2, 'vlm_model': server.vlm_api.DASHSCOPE_MODEL, 'vlm_provider': config.VLM_PROVIDER,
          'key_recorded': False, 'runtime_results_isolated': True})
    started = time.monotonic(); done = len(prior)
    try:
        for index, sample in enumerate(p['samples']):
            destination = output/'cases'/f'{index:05d}.json'
            if destination.exists(): continue
            if calls[0] >= p['maximum_vlm_calls'] or failures[0] >= 3:
                raise RuntimeError('Regression stopped after VLM budget/three consecutive failures; retain partial report')
            path = ROOT/sample['path']
            if file_hash(path) != sample['file_sha256'] or image_hash(path) != sample['pixel_hash']:
                raise ValueError('Source image changed')
            token = BATCH_CONTEXT.set({'record_scope': 'evaluation', 'regression_run': p['version'], 'regression_case': index})
            wall = time.monotonic()
            try:
                result = inspect(path.read_bytes(), sample['pixel_hash'][:16]+path.suffix, 'screw')
                bound = result.get('quality_model_bound',False); view = (result.get('family_routing') or {}).get('view_key')
                applied_wrong = bool(bound and (sample['expected_view'] is None or view != sample['expected_view']))
                valid = bool(bound and not applied_wrong and sample['truth'] is not None and result.get('perception_prediction') in ('normal','defect'))
                row = {'sample': sample, 'status': 'completed' if result.get('ok') else 'error',
                       'valid_quality_prediction': valid, 'misapplied_quality_model': applied_wrong,
                       'wall_seconds': round(time.monotonic()-wall,4),
                       'result': {k:v for k,v in result.items() if k not in ('heatmap','classification')}}
            except Exception as exc:
                row = {'sample': sample, 'status': 'error', 'valid_quality_prediction': False,
                       'wall_seconds': round(time.monotonic()-wall,4), 'error': str(exc), 'result': {}}
            finally: BATCH_CONTEXT.reset(token)
            write(destination, row); done += 1
            if done % 25 == 0 or done == len(p['samples']):
                write(output/'progress.json', {'completed':done,'total':len(p['samples']),'vlm_calls':calls[0],'seconds_this_session':round(time.monotonic()-started)})
                print({'completed': done, 'total': len(p['samples']), 'cohort': sample['cohort'], 'vlm_calls': calls[0], 'seconds':round(time.monotonic()-started)},flush=True)
        verify(output)
    finally:
        result = summarize(output)
        print({'processed': result['processed'], 'complete': result['complete']},flush=True)
        if server._review_workflow: server._review_workflow.executor.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','run','summarize']); parser.add_argument('--output',type=Path,default=OUTPUT)
    args = parser.parse_args(); {'prepare':prepare,'run':run,'summarize':summarize}[args.action](args.output.resolve())


if __name__ == '__main__': main()
