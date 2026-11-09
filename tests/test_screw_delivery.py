import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from products import screw_delivery as delivery
from web import screw_delivery_api


class ScrewDeliveryTests(unittest.TestCase):
    def test_unverified_acceptance_cannot_be_displayed_as_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); target = root/'product_families/screw/delivery_acceptance.json'
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps({'schema': 1, 'passed': True, 'record_scope': 'production'}), encoding='utf8')
            with patch.object(delivery, 'runtime_inventory', return_value={}), patch.object(delivery, 'read_evidence', return_value=None):
                with self.assertRaises(ValueError):
                    delivery.showcase(root, [], {}, {})
    def test_metrics_reject_inconsistent_percentages_instead_of_hiding_errors(self):
        m = delivery.checked_metrics({'n': 38, 'confusion_matrix': [[19, 0], [3, 16]]})
        self.assertEqual(m['false_negatives'], 3)
        self.assertAlmostEqual(m['recall'], 16/19)
        for invalid in ({'accuracy': 1.}, {'n': 39}, {'confusion_matrix': [[19, 0], [-1, 16]]}):
            with self.assertRaises(ValueError):
                delivery.checked_metrics({'n': 38, 'confusion_matrix': [[19, 0], [3, 16]], **invalid})

    def test_old_result_is_not_marked_current_after_model_or_threshold_changes(self):
        inventory = {'view_groups': [{'key': 'head', 'model_arrays_sha256': 'same', 'threshold': .2, 't_high': .8}],
                     'router_arrays_sha256': 'router', 'appearance_arrays_sha256': 'appearance'}
        evidence = {'quality_cohorts': [{'view': 'head', 'runtime_model_arrays_sha256': 'same', 'runtime_threshold': .3, 'runtime_t_high': .8}],
                    'routing': {'arrays_sha256': 'router'}, 'appearance': {'arrays_sha256': 'appearance'}}
        with patch.object(delivery, 'runtime_inventory', return_value=inventory), patch.object(delivery, 'read_evidence', return_value=evidence):
            result = delivery.showcase('unused', [], {}, {})
        self.assertFalse(result['evidence']['quality_cohorts'][0]['matches_current_model'])
        self.assertTrue(result['evidence']['routing_matches_current'])

    def test_absent_evidence_does_not_generate_example_accuracy(self):
        with patch.object(delivery, 'runtime_inventory', return_value={}), patch.object(delivery, 'read_evidence', return_value=None):
            result = delivery.showcase('unused', [], {}, {})
        self.assertFalse(result['evidence_ready'])
        self.assertIsNone(result['evidence'])

    def test_export_keeps_failed_candidate_and_evaluation_scope(self):
        app = FastAPI(); app.include_router(screw_delivery_api.router); client = TestClient(app)
        evidence = {'quality_cohorts': [{'name': 'research-only', 'population': '38 source specimens',
                    'metrics': delivery.checked_metrics({'confusion_matrix': [[19, 0], [3, 16]]}),
                    'matches_current_model': False, 'state': 'not released', 'scope': 'no VLM or human correction'}]}
        with patch.object(screw_delivery_api, 'snapshot', return_value={'evidence_ready': True, 'evidence': evidence}):
            response = client.get('/api/product-families/screw/showcase/export')
        self.assertEqual(response.status_code, 200)
        self.assertIn('not released', response.text)
        self.assertIn('no VLM or human correction', response.text)
        self.assertIn('text/csv', response.headers['content-type'])
        with patch.object(screw_delivery_api, 'snapshot', return_value={'evidence_ready': False}):
            self.assertEqual(client.get('/api/product-families/screw/showcase/export').status_code, 409)


if __name__ == '__main__':
    unittest.main()
