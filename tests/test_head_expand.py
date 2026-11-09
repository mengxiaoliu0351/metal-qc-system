import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from experiments import m2_head_expand as experiment
from experiments.m2_head_expand import KEY, no_regression, select_training
from experiments.prepare_m2_body import sha
from products.families import read, write


class HeadExpansionTests(unittest.TestCase):
    def test_training_excludes_every_reserved_role_and_publisher_test(self):
        rows=[{'role':role,'pixel_hash':role} for role in ('train','feedback','test','calibration','qualification')]
        rows += [{'role':'feedback','pixel_hash':'reserved_global'}, {'role':'feedback','pixel_hash':'reserved_product'}]
        reserve={'global':{'reserved_global':{}},'products':{KEY:{'reserved_product':{}}}}
        selected=select_training(rows,reserve)
        self.assertEqual({s['pixel_hash'] for s in selected},{'train','feedback'})
        self.assertTrue(all(s['role']=='train' for s in selected))

    def test_accuracy_gain_cannot_hide_recall_or_false_positive_regression(self):
        old={'accuracy':.91,'balanced_accuracy':.91,'recall':.94,'false_positive_rate':.12}
        better={**old,'accuracy':.93,'balanced_accuracy':.93,'false_positive_rate':.08}
        self.assertTrue(no_regression(old,better))
        self.assertFalse(no_regression(old,{**better,'recall':.93}))
        self.assertFalse(no_regression(old,{**better,'false_positive_rate':.13}))

    def test_failed_calibration_stops_before_reading_historical_feature_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder);p={'version':'fixture','scope':'historical'}
            write(output/'protocol.json',p);np.savez(output/'candidate.npz')
            write(output/'candidate.json',{'arrays_sha256':sha(output/'candidate.npz'),
                  'protocol_sha256':sha(output/'protocol.json'),'development_passed':False})
            with patch.object(experiment,'verify',return_value=p), \
                 patch('experiments.screw_family.Features',side_effect=AssertionError('Historical feature cache opened')):
                experiment.evaluate(output)
            report=read(output/'report.json')
            self.assertEqual(report['cohorts'],{})
            self.assertFalse(report['release_permission'])
            self.assertFalse(report['automatic_archival'])


if __name__=='__main__':unittest.main()
