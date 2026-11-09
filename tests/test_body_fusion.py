from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments import m2_body_fusion as fusion
from experiments.prepare_m2_body import sha
from products.families import read, write


class BodyFusionTests(unittest.TestCase):
    def test_selection_reads_calibration_only(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            p = {'version': 'fixture', 'detail_weights': [0, .25, .5, 1],
                 'target': {'minimum_recall': .93, 'maximum_false_positive_rate': .08}}
            write(output/'protocol.json', p)
            y = np.array([0]*20+[1]*20)
            base = np.r_[np.linspace(.05, .3, 20), np.linspace(.4, .9, 20)]
            def inputs(role):
                self.assertEqual(role, 'calibration')
                return [], y, base, base, np.ones(len(y), bool)
            with patch.object(fusion, 'verify', return_value=p), patch.object(fusion, 'inputs', side_effect=inputs):
                fusion.fit(output)
            self.assertEqual(read(output/'model.json')['detail_weight'], 0)
            self.assertFalse(read(output/'development_report.json')['historical_cohorts_read'])

    def test_historical_success_never_grants_production_permission(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder); prior = output/'detail'; prior.mkdir()
            p = {'version': 'fixture'}; write(output/'protocol.json', p)
            write(output/'model.json', {'detail_weight': .5, 'thresholds': [.5, .8],
                                       'protocol_sha256': sha(output/'protocol.json')})
            write(prior/'report.json', {'cohorts': {role: {'baseline_v2': {}, 'detail_candidate': {}}
                                                  for role in ('qualification', 'test', 'regression')}})
            y = np.array([0]*20+[1]*20); score = np.r_[np.full(20, .1), np.full(20, .9)]
            rows = [{'pixel_hash': str(i), 'label': 'defect' if v else 'normal'} for i, v in enumerate(y)]
            with patch.object(fusion, 'verify', return_value=p), patch.object(fusion.detail, 'OUTPUT', prior), \
                    patch.object(fusion, 'inputs', return_value=(rows, y, score, score, np.ones(40, bool))):
                fusion.evaluate(output)
            r = read(output/'report.json')
            self.assertTrue(r['all_historical_binary_gates_passed'])
            self.assertFalse(r['eligible_for_production'])
            self.assertFalse(r['automatic_archival'])
            self.assertFalse(r['new_blind_test'])


if __name__ == '__main__':
    unittest.main()
