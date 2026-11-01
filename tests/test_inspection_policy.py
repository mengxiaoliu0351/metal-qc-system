"""Selective actions, human truth and persistence; all writes are isolated."""
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from review.disposition import InspectionPolicy, effective_disposition, image_quality, validate_references
from review.workflow import ReviewWorkflow, read, write
from review.metrics import operational_summary
from interface import report
from web import server


class SelectivePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.results = self.root / 'results'
        self.workflow = ReviewWorkflow(self.root, self.results)
        self.policy = self.workflow.policy()
        self.config = {'key': 'metal_nut', 'threshold': .3, 't_high': .7}
        self.profile = {'model_key': 'version1', 'auto_low': .2, 'auto_high': .8,
                        'allow_normal': True, 'allow_isolate': True}
        self.policy.profile = Mock(return_value=self.profile)
        self.policy.scope_check = Mock(return_value={'ok': True})

    def tearDown(self):
        self.workflow.executor.shutdown(wait=True)
        self.temp.cleanup()

    def record(self, score=.1, route='low_risk', fingerprint='fixture'):
        return {'session_id': 'sample', 'product_key': 'metal_nut', 'image_hash': fingerprint,
                'perception_prediction': 'normal' if score < .3 else 'defect', 'score': score,
                'quality_check': {'ok': True}, 'gate': {'route': route}, 'cognition': {'ok': True}}

    def decide(self, record):
        return self.policy.decide(record, self.config, np.ones(4))

    def test_auto_actions_and_stable_audit_selection(self):
        found = {}
        for n in range(300):
            rec = self.record(fingerprint=str(n))
            action = self.decide(rec)
            found.setdefault(action['sampled'], (rec, action))
            self.assertEqual(action, self.decide(rec))
        self.assertEqual(set(found), {False, True})
        self.assertEqual(found[False][1]['status'], 'auto_normal')
        self.assertEqual(found[True][1]['status'], 'sampled_review')
        self.assertTrue(found[True][1]['requires_review'])
        self.assertEqual(found[True][1]['auto_result'], 'normal')
        high = self.decide(self.record(.95, 'high_risk'))
        self.assertEqual(high['system_action'], 'isolate')
        self.assertEqual(high['auto_result'], 'defect')
        self.assertNotIn('human_verdict', high)
        self.assertEqual(self.workflow.approved('metal_nut'), [])

    def test_boundaries_failed_qualification_and_invalid_score_need_review(self):
        for score, route in ((.2, 'low_risk'), (.8, 'high_risk'), (.25, 'low_risk'), (float('nan'), 'high_risk')):
            self.assertTrue(self.decide(self.record(score, route))['requires_review'])
        self.profile['allow_normal'] = False
        self.assertEqual(self.decide(self.record())['reason_code'], 'guard_band')
        self.policy.profile.side_effect = FileNotFoundError()
        self.assertEqual(self.decide(self.record())['reason_code'], 'unvalidated_model')

    def test_gray_failures_conflicts_and_unqualified_evidence_are_manual(self):
        for cognition, reason in (({'ok': False}, 'vlm_failed'),
            ({'ok': True, 'citation_check': {'ok': False}}, 'invalid_citation'),
            ({'ok': True, 'vlm_conclusion': '正常'}, 'evidence_conflict'),
            ({'ok': True}, 'gray_review')):
            rec = self.record(.5, 'vlm_review'); rec['cognition'] = cognition
            self.assertEqual(self.decide(rec)['reason_code'], reason)

    def test_quality_scope_unknown_product_and_validation_mode(self):
        rec = self.record(); rec['quality_check']['ok'] = False
        self.assertEqual(self.decide(rec)['reason_code'], 'input_quality')
        self.policy.scope_check.return_value = {'ok': False, 'reason': 'wrong product'}
        self.assertEqual(self.decide(self.record())['reason_code'], 'scope_mismatch')
        self.assertEqual(self.policy.decide(self.record(), None)['reason_code'], 'unregistered_product')
        self.policy.configure({'mode': 'validation', 'sample_rate': .1, 'operator': 'fixture'})
        self.assertEqual(self.decide(self.record())['reason_code'], 'validation_mode')
        self.assertEqual(self.decide(self.record(.9, 'processing_error'))['reason_code'], 'processing_error')
        for rate in (float('nan'), True, 0, 1.01):
            with self.assertRaises(ValueError): self.policy.configure({'mode': 'assisted', 'sample_rate': rate, 'operator': 'fixture'})

    def test_human_disagreement_suspends_version_without_training(self):
        rec = self.record(); rec['disposition'] = self.decide(rec)
        path = self.results / 'history/2026-10-26/sample.json'; write(path, rec)
        response = self.workflow.confirm({'session_id': 'sample', 'verdict': 'defect', 'reviewer': 'fixture', 'review_seconds': 8.5})
        self.assertTrue(response['automation_suspended'])
        self.assertFalse(response['learned'])
        self.assertEqual(self.decide(self.record())['reason_code'], 'quality_hold')
        self.assertEqual(effective_disposition(read(path))['status'], 'human_confirmed')
        self.assertEqual(self.workflow.queue(status='pending')['count'], 0)
        self.assertEqual(self.workflow.approved('metal_nut'), [])
        self.profile['model_key'] = 'version2'
        self.assertIn(self.decide(self.record())['status'], ('auto_normal', 'sampled_review'))

    def test_legacy_is_not_retroactively_processed_and_reopened_labels_are_excluded(self):
        legacy = self.record(); write(self.results / 'history/2026-10-26/old.json', legacy)
        self.assertEqual(effective_disposition(legacy)['status'], 'legacy')
        self.assertEqual(self.workflow.queue(status='pending')['count'], 0)
        self.assertEqual(self.workflow.queue(status='legacy')['count'], 1)
        image = self.root / 'sample.png'; Image.new('RGB', (128, 128)).save(image)
        legacy['input_png'] = str(image); write(self.results / 'history/2026-10-26/old.json', legacy)
        self.workflow.confirm({'session_id': 'sample', 'verdict': 'ok', 'reviewer': 'fixture', 'use_for_iteration': True})
        self.assertEqual(len(self.workflow.approved('metal_nut')), 1)
        saved = read(self.results / 'history/2026-10-26/old.json'); saved['review_reopened'] = True
        write(self.results / 'history/2026-10-26/old.json', saved)
        self.assertEqual(self.workflow.approved('metal_nut'), [])
        self.assertEqual(self.workflow.queue(status='pending')['count'], 1)

    def test_quality_metrics_use_only_confirmed_truth(self):
        automatic = self.record(); automatic['disposition'] = self.decide(automatic); automatic['time_total'] = 2
        labelled = copy.deepcopy(automatic)
        labelled['human_verdict'] = {'confirmed': True, 'verdict': 'defect', 'review_seconds': 10}
        reopened = copy.deepcopy(labelled); reopened['review_reopened'] = True
        metrics = operational_summary([automatic, labelled, reopened, {}])
        self.assertEqual(metrics['new_policy_records'], 3)
        self.assertEqual(metrics['labelled_comparisons'], 1)
        self.assertEqual(metrics['false_negative'], 1)
        self.assertEqual(metrics['reviewed_agreement'], 0)
        self.assertEqual(metrics['review_median_seconds'], 10)

    def test_reference_ids_and_image_quality(self):
        self.assertFalse(image_quality(Image.new('RGB', (300, 300)))['ok'])
        self.assertFalse(validate_references({'ref_ids': ['fake']}, [{'id': 'real'}])['citation_check']['ok'])
        self.assertFalse(validate_references({'ref_ids': 'real'}, [{'id': 'real'}])['citation_check']['ok'])
        self.assertTrue(validate_references({'ref_ids': ['real']}, [{'id': 'real'}])['citation_check']['ok'])


class PersistenceContractTests(unittest.TestCase):
    def test_calibration_manifest_does_not_depend_on_uploaded_image_subset(self):
        import review.disposition as module
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); results = root / 'results'
            split = {'validation': ['data/good/v.png', 'data/scratch/v.png'],
                     'normal_calibration': ['data/good/cal.png'],
                     'test': ['data/good/t.png', 'data/scratch/t.png']}
            signature = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
            base = results / 'experiments/perception_upgrade_v1'
            write(base / 'metal_nut_split.json', split)
            paths = split['validation'] + split['normal_calibration'] + split['test']
            np.savez(base / 'metal_nut_features.npz', signature=np.array(signature), paths=np.array(paths),
                     global_features=np.array([[-2.,0.],[2.,0.],[-1.,0.],[-2.,0.],[2.,0.]]), normal_global=np.array([[-2.,0.]]))
            optimized = results / 'perception/optimized'; optimized.mkdir(parents=True)
            np.savez(optimized / 'test.npz', global_mean=np.zeros(2), global_scale=np.ones(2), global_coef=np.array([[1.,0.]]), global_intercept=np.zeros(1))
            write(optimized / 'test.json', {'product': 'metal_nut', 'mode': 'global_binary', 'threshold': .4, 'arrays': 'test.npz', 'normal_calibration_count': 1})
            write(root / 'calibration_exclusions.json', {'metal_nut': {'split_signature': signature, 'legacy_memory': []}})
            config = {'key': 'metal_nut', 'perception_artifact': 'test.json', 'threshold': .4, 't_high': .7}
            with patch.object(module, '__file__', str(root / 'disposition.py')):
                first = InspectionPolicy(root, results).profile(config)
                (root / 'data/mvtec/metal_nut/train/good').mkdir(parents=True)
                Image.new('RGB', (2, 2)).save(root / 'data/mvtec/metal_nut/train/good/new.png')
                second = InspectionPolicy(root, results).profile(config)
                self.assertEqual(first, second)
                write(root / 'calibration_exclusions.json', {'metal_nut': {'split_signature': 'changed', 'legacy_memory': []}})
                with self.assertRaises(ValueError): InspectionPolicy(root, results).profile(config)

    def test_valid_inputs_and_failures_are_archived_without_false_normal_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); results = root / 'results'
            workflow = ReviewWorkflow(root, results)
            client = TestClient(server.app)
            good = Image.fromarray(np.random.default_rng(2).integers(0, 256, (128, 128, 3), dtype=np.uint8))
            def detect(image, product):
                raw = io.BytesIO(); image.save(raw, format='PNG')
                return client.post('/api/detect', files={'file': ('same.png', raw.getvalue(), 'image/png')}, data={'product': product}).json()
            try:
                with patch.object(server, 'RESULTS_DIR', str(results)), patch.object(report, 'HISTORY_DIR', str(results / 'history')), patch.object(server, 'get_review_workflow', return_value=workflow), patch.object(server.vlm_api, 'analyze', return_value={'ok': False, 'error': 'test failure'}) as vlm, patch.object(server, 'get_engine', side_effect=RuntimeError('fixture')):
                    for image, product, reason in ((Image.new('RGB', (128, 128)), 'metal_nut', 'input_quality'), (good, 'vlm_general', 'unregistered_product'), (good, 'metal_nut', 'processing_error')):
                        response = detect(image, product)
                        self.assertTrue(response['ok'], response)
                        self.assertEqual(response['disposition']['reason_code'], reason)
                        self.assertIsNone(response['score'])
                        self.assertTrue(response['session_id'])
                    self.assertEqual(vlm.call_count, 1)
                    records = list(workflow.records())
                    self.assertEqual(len(records), 3)
                    self.assertEqual(workflow.queue(status='pending')['count'], 3)
                    self.assertTrue(all(Path(r['input_png']).exists() for _, r in records))
                    self.assertTrue(all(r.get('perception_prediction') is None for _, r in records))
                    self.assertEqual(workflow.approved('metal_nut'), [])
                    self.assertEqual(len({p.name for p, _ in records}), 3)
            finally: workflow.executor.shutdown(wait=True)


if __name__ == '__main__': unittest.main()
