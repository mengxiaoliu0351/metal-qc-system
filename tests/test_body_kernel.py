from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments import m2_body_kernel as experiment
from perception.body_kernel import export_kernel, kernel_margin
from products.families import read, write


class BodyKernelTests(unittest.TestCase):
    def test_portable_prediction_matches_fitted_svm(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        rng = np.random.default_rng(19)
        x = rng.normal(size=(60, 5)); y = (x[:, 0]*x[:, 1] > 0).astype(int)
        model = Pipeline([('sc', StandardScaler()), ('clf', SVC(C=.1, gamma='scale'))]).fit(x, y)
        query = np.r_[rng.normal(size=(12, 5)), np.zeros((1,5)), np.ones((1,5))*100]
        np.testing.assert_allclose(kernel_margin(query, export_kernel(model)), model.decision_function(query), atol=1e-10)

    def test_training_never_reads_heldout_features_and_refuses_refit(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            samples = [{'role':role, 'label':'defect' if i%2 else 'normal', 'pixel_hash':f'{role}_{i}'}
                       for role,n in [('train',40),('calibration',20),('qualification',2),('test',2),('regression',2)] for i in range(n)]
            p = {'version':'fixture', 'samples':samples, 'classifier':{'kernel':'rbf','C':.1,'gamma':'scale','class_weight':'balanced','probability':False},
                 'seed':1,'calibration_target':{'minimum_recall':.93,'maximum_false_positive_rate':.08}}
            write(output/'protocol.json', p)
            def features(_, rows, variant='original'):
                self.assertTrue(all(s['role'] in ('train','calibration') for s in rows))
                return np.array([[1.,float(s['label']=='defect')] for s in rows])
            with patch.object(experiment,'verify',return_value=p),patch.object(experiment,'vectors',side_effect=features):
                experiment.fit(output)
                with self.assertRaisesRegex(ValueError,'refit'): experiment.fit(output)
            self.assertFalse(read(output/'development_report.json')['heldout_features_read'])
            self.assertFalse(read(output/'candidate.json')['release_permission'])

    def test_failed_development_stops_before_historical_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder); write(output/'protocol.json', {})
            p = {'version':'fixture','evaluation_scope':'historical'}
            with patch.object(experiment,'candidate',return_value=(p,{'development_target_met':False},{})), \
                 patch.object(experiment,'verify',return_value=p), \
                 patch.object(experiment,'cohort',side_effect=AssertionError('Heldout read after failed development')):
                experiment.evaluate(output)
            report = read(output/'report.json')
            self.assertFalse(report['release_permission']); self.assertFalse(report['historical_numerical_gates_passed'])
            self.assertEqual(report['cohorts'], {})


if __name__ == '__main__': unittest.main()
