"""Isolated acceptance check with real frozen data; never publish or add production labels."""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, '/app/src')
from review.workflow import ReviewWorkflow, read, write
from web.server import get_review_workflow

root = Path('/app')
production = root / 'results/perception/banks/products.json'
before = hashlib.sha256(production.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='metalqc-acceptance-') as temp:
    results = Path(temp)
    shutil.copytree(root / 'results/perception', results / 'perception')
    (results / 'experiments').mkdir()
    (results / 'experiments/perception_upgrade_v1').symlink_to(root / 'results/experiments/perception_upgrade_v1', target_is_directory=True)
    workflow = ReviewWorkflow(root, results, extractor=get_review_workflow().extractor)
    split = read(workflow.benchmark / 'metal_nut_split.json')
    sample = root / split['train'][0].replace('\\', '/')
    verdict = 'ok' if sample.parent.name == 'good' else 'defect'
    write(results / 'history/2026-10-25/acceptance.json', {
        'session_id': 'acceptance-only', 'product_key': 'metal_nut', 'input_png': str(sample),
        'sample_name': sample.name, 'perception_prediction': 'normal', 'score': .5})
    workflow.confirm({'session_id': 'acceptance-only', 'verdict': verdict,
                      'reviewer': '部署自动验收（公开数据集标签）', 'use_for_iteration': True})
    run = workflow.start('metal_nut', 'model')
    workflow.futures[run['id']].result(timeout=180)
    state = read(workflow.run_path(run['id']) / 'state.json')
    assert state['status'] == 'completed', state.get('message')
    vectors, baseline = workflow.extractor('metal_nut', [str(sample)])
    assert vectors.shape == (1, 512) and baseline.shape == (1,)
    assert np.isfinite(vectors).all() and np.isfinite(baseline).all()
    workflow.executor.shutdown(wait=True)
    receipt = {'candidate_evaluated': True, 'real_shared_feature_extraction': True,
               'eligible': state['report']['eligible_for_activation'],
               'production_unchanged': hashlib.sha256(production.read_bytes()).hexdigest() == before,
               'production_labels_added': 0, 'published': False}
    assert receipt['production_unchanged']
    (root / 'results/deployment_checks/iteration-unified.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
