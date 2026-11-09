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
from review.workflow import ReviewWorkflow, original_prediction, read, write
from review.metrics import operational_summary, gray_decision_summary, audit_window_summary
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

    def test_sealed_evaluation_errors_veto_without_training_or_clearing_existing_holds(self):
        batch = {"id": "sealed", "purpose": "evaluation", "status": "completed",
                 "recipe": {"config": self.config}, "items": [
                     {"id": "bad", "status": "completed", "dataset_truth": "defect",
                      "result": {"disposition": {"model_key": "version1", "auto_result": "normal"}}}]}
        result = self.policy.observe_evaluation(batch)
        self.assertTrue(result["automatic_release_vetoed"])
        self.assertEqual(self.decide(self.record())["reason_code"], "quality_hold")
        self.assertEqual(self.workflow.approved('metal_nut'), [])
        batch["items"] = []
        self.policy.observe_evaluation(batch)
        self.assertEqual(self.decide(self.record())["reason_code"], "quality_hold")
        batch["purpose"] = "production"
        with self.assertRaises(ValueError): self.policy.observe_evaluation(batch)

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

    def test_gray_clear_results_archive_and_ambiguous_results_wait_for_review(self):
        normal = self.record(.31, 'vlm_review')
        normal['cognition'] = {'ok': True, 'vlm_conclusion': '合格', 'conclusion': '合格',
            'defect_type': '正常', 'defect_desc': '未见明显缺陷',
            'citation_check': {'ok': True, 'has_citations': True}}
        normal_action = self.decide(normal)
        self.assertEqual(normal_action['auto_result'], 'normal')
        self.assertEqual(normal_action['decision_source'], 'gray_vlm_rag')
        self.assertIn(normal_action['status'], ('auto_normal', 'sampled_review'))
        defect = self.record(.55, 'vlm_review')
        defect['cognition'] = {'ok': True, 'conclusion': '不合格', 'defect_type': '划痕',
            'defect_desc': '外缘有线状划痕', 'citation_check': {'ok': True, 'has_citations': True}}
        defect_action = self.decide(defect)
        self.assertEqual(defect_action['auto_result'], 'defect')
        self.assertEqual(defect_action['system_action'], 'isolate')
        self.assertEqual(defect_action['decision_source'], 'gray_vlm_rag')
        self.assertIn(defect_action['status'], ('auto_isolated', 'sampled_review'))
        self.profile['allow_normal'] = False
        self.assertEqual(self.decide(normal)['reason_code'], 'unvalidated_normal')
        self.profile['allow_isolate'] = False
        self.assertEqual(self.decide(defect)['reason_code'], 'unvalidated_isolate')

    def test_gray_failures_conflicts_and_unqualified_evidence_are_manual(self):
        for cognition, reason in (({'ok': False}, 'vlm_failed'),
            ({'ok': True, 'citation_check': {'ok': False}}, 'invalid_citation'),
            ({'ok': True, 'vlm_conclusion': '正常'}, 'invalid_citation'),
            ({'ok': True, 'conclusion': '需复核', 'citation_check': {'ok': True, 'has_citations': True}}, 'gray_uncertain'),
            ({'ok': True, 'conclusion': '不合格', 'defect_type': '自造缺陷', 'defect_desc': '未知',
              'citation_check': {'ok': True, 'has_citations': True}}, 'gray_uncertain')):
            rec = self.record(.5, 'vlm_review'); rec['cognition'] = cognition
            self.assertEqual(self.decide(rec)['reason_code'], reason)
            self.assertIsNone(self.decide(rec)['auto_result'])

    def test_quality_scope_unknown_product_and_reject_retired_mode(self):
        rec = self.record(); rec['quality_check']['ok'] = False
        self.assertEqual(self.decide(rec)['reason_code'], 'input_quality')
        self.policy.scope_check.return_value = {'ok': False, 'reason': 'wrong product'}
        self.assertEqual(self.decide(self.record())['reason_code'], 'scope_mismatch')
        self.assertEqual(self.policy.decide(self.record(), None)['reason_code'], 'unregistered_product')
        with self.assertRaises(ValueError):
            self.policy.configure({'mode': 'validation', 'sample_rate': .1, 'operator': 'fixture'})
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

    def test_gray_and_period_audit_metrics_keep_machine_and_human_separate(self):
        self.policy.configure({'mode': 'assisted', 'sample_rate': .2,
                               'audit_period': 'week', 'operator': 'fixture'})
        rec = self.record(.55, 'vlm_review')
        rec['timestamp'] = '2026-11-03T12:00:00'
        rec['cognition'] = {'ok': True, 'conclusion': '不合格', 'defect_type': '划痕',
            'defect_desc': '外缘划痕', 'citation_check': {'ok': True, 'has_citations': True}}
        rec['disposition'] = self.decide(rec)
        self.assertEqual(rec['disposition']['audit_period_key'], '2026-W45')
        self.assertEqual(rec['disposition']['auto_result'], 'defect')
        self.assertEqual(self.decide(rec)['sampled'], rec['disposition']['sampled'])
        rec['human_verdict'] = {'confirmed': True, 'verdict': 'ok'}
        duplicate = copy.deepcopy(rec)
        gray = gray_decision_summary([rec, duplicate])
        self.assertEqual(gray['total'], 1)
        self.assertEqual(gray['defect_false_positives'], 1)
        audit = audit_window_summary([rec], '2026-W45')
        self.assertEqual(audit['automatic'], 1)
        self.assertEqual(audit['selected'], int(rec['disposition']['sampled']))
        self.assertEqual(original_prediction(rec), 'defect')

    def test_reference_ids_and_image_quality(self):
        self.assertFalse(image_quality(Image.new('RGB', (300, 300)))['ok'])
        self.assertFalse(validate_references({'ref_ids': ['fake']}, [{'id': 'real'}])['citation_check']['ok'])
        self.assertFalse(validate_references({'ref_ids': 'real'}, [{'id': 'real'}])['citation_check']['ok'])
        self.assertTrue(validate_references({'ref_ids': ['real']}, [{'id': 'real'}])['citation_check']['ok'])


class PersistenceContractTests(unittest.TestCase):
    def test_gray_detection_archives_clear_results_and_queues_uncertain_with_periodic_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); results = root / 'results'
            workflow = ReviewWorkflow(root, results)
            policy = workflow.policy()
            policy.profile = Mock(return_value={'model_key': 'fixture-v1', 'auto_low': .2, 'auto_high': .8,
                                                'allow_normal': True, 'allow_isolate': True})
            policy.scope_check = Mock(return_value={'ok': True})
            policy.configure({'mode': 'assisted', 'sample_rate': 1.0, 'audit_period': 'day', 'operator': 'fixture'})
            config = {'key': 'metal_nut', 'name': 'test product', 'threshold': .3, 't_high': .7}
            images = [Image.fromarray(np.random.default_rng(3 + i).integers(0, 256, (128, 128, 3), dtype=np.uint8))
                      for i in range(3)]
            image_bytes = []
            for image in images:
                raw = io.BytesIO(); image.save(raw, format='PNG'); image_bytes.append(raw.getvalue())
            engine = Mock()
            engine.detect.side_effect = [{'score': score, 'heatmap': images[i], 'classification': {}}
                                         for i, score in enumerate((.31, .55, .45))]
            analyses = [
                {'ok': True, 'conclusion': '合格', 'defect_type': '正常',
                 'defect_desc': '表面未见异常', 'ref_ids': ['fixture-ref']},
                {'ok': True, 'conclusion': '不合格', 'defect_type': '划痕',
                 'defect_desc': '外缘线状划痕', 'ref_ids': ['fixture-ref']},
                {'ok': True, 'conclusion': '需复核', 'defect_type': '其他',
                 'defect_desc': '证据不确定', 'ref_ids': ['fixture-ref']}]
            kb = Mock()
            kb.retrieve.return_value = [{'id': 'fixture-ref', 'standard': 'fixture',
                                         'section': '1', 'text': 'fixture text'}]
            try:
                with patch.object(server, 'RESULTS_DIR', str(results)), \
                     patch.object(report, 'HISTORY_DIR', str(results / 'history')), \
                     patch.object(server, 'get_review_workflow', return_value=workflow), \
                     patch.object(server, 'get_engine', return_value=(engine, config)), \
                     patch.object(server, 'get_clf', return_value=Mock(available=False)), \
                     patch.object(server, 'get_sem_kb', return_value=kb), \
                     patch.object(server.vlm_api, 'analyze', side_effect=analyses), \
                     patch.object(server, 'apply_optimized_perception',
                                  side_effect=lambda _product, _cfg, _engine, _image, result:
                                      {**result, 'image_features': np.ones(4)}):
                    client = TestClient(server.app)
                    outcomes = [client.post('/api/detect', files={'file': (f'{i}.png', image_bytes[i], 'image/png')},
                                            data={'product': 'metal_nut'}).json() for i in range(3)]
                    self.assertTrue(all(x['ok'] for x in outcomes), outcomes)
                    self.assertEqual([x['disposition']['auto_result'] for x in outcomes],
                                     ['normal', 'defect', None])
                    self.assertEqual([x['disposition']['status'] for x in outcomes],
                                     ['sampled_review', 'sampled_review', 'needs_review'])
                    self.assertEqual(client.get('/api/review/queue', params={'status': 'sampled'}).json()['count'], 2)
                    self.assertEqual(client.get('/api/review/queue', params={'status': 'gray'}).json()['count'], 3)
                    audit = client.get('/api/inspection/audit').json()
                    self.assertEqual((audit['automatic'], audit['selected'], audit['selected_pending']), (2, 2, 2))
                    gray = client.get('/api/inspection/gray-evidence').json()
                    self.assertEqual((gray['total'], gray['auto_normal'], gray['auto_defect'], gray['uncertain']),
                                     (3, 1, 1, 1))
                    review = client.post('/api/review/confirm', json={'session_id': outcomes[0]['session_id'],
                        'verdict': 'defect', 'reviewer': 'fixture'}).json()
                    self.assertTrue(review['automation_suspended'])
                    audit_after = client.get('/api/inspection/audit').json()
                    self.assertEqual((audit_after['selected_reviewed'], audit_after['confirmed_errors']), (1, 1))
                    self.assertEqual(client.get('/api/inspection/gray-evidence').json()['normal_misses'], 1)
                    self.assertEqual(client.get('/api/review/queue', params={'status': 'sampled'}).json()['count'], 1)
                    self.assertEqual(workflow.approved('metal_nut'), [])
            finally:
                workflow.executor.shutdown(wait=True)

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
