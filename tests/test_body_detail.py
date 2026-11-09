from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from experiments import m2_body_detail as detail
from perception.body_detail import prepare_detail, detail_features
from products.families import read, write


class DetailBodyTests(unittest.TestCase):
    def test_disconnected_shaft_and_head_stay_in_one_proposal(self):
        image = Image.new('RGB', (400, 500), 'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((190, 180, 210, 265), fill=(50, 50, 50))
        draw.rectangle((183, 273, 217, 289), fill=(50, 50, 50))
        draw.rectangle((0, 330, 399, 499), fill=(40, 40, 40))
        prepared, meta = prepare_detail(image)
        self.assertIsNotNone(prepared)
        box = meta['source_box']
        self.assertLessEqual(box[1], 180)
        self.assertGreaterEqual(box[3], 289)
        self.assertLess(box[3], 330)
        self.assertEqual(prepared.size, (96, 256))
        vector, _ = detail_features(image)
        self.assertEqual(vector.shape, (12276,))
        self.assertTrue(np.isfinite(vector).all())

    def test_blank_and_table_only_do_not_become_screw_proposals(self):
        for table in (False, True):
            image = Image.new('RGB', (400, 500), 'white')
            if table:
                ImageDraw.Draw(image).rectangle((0, 300, 399, 499), fill=(40, 40, 40))
            prepared, meta = prepare_detail(image)
            self.assertIsNone(prepared)
            self.assertFalse(meta['proposal_found'])

    def test_fitting_cannot_read_historical_holdout_features_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'features').mkdir()
            samples = []
            for role, count in [('train', 30), ('calibration', 20), ('qualification', 2), ('test', 2), ('regression', 2)]:
                for i in range(count):
                    sample = {'pixel_hash': f'{role}_{i}', 'role': role, 'label': 'defect' if i%2 else 'normal'}
                    samples.append(sample)
                    if role in ('train', 'calibration'):
                        np.savez_compressed(root/'features'/(sample['pixel_hash']+'.npz'),
                                            detail=np.array([i%2, i/100]), proposal_found=np.array(True))
            protocol = {'version': 'fixture', 'samples': samples, 'modes': ['detail_hog'], 'classifier_C': [.01, .1],
                        'seed': 1, 'calibration_target': {'minimum_recall': .93, 'maximum_false_positive_rate': .08}}
            write(root/'protocol.json', protocol)
            with patch.object(detail, 'verify', return_value=protocol):
                detail.fit(root)
                with self.assertRaisesRegex(ValueError, 'refit'):
                    detail.fit(root)
            report = read(root/'development_report.json')
            self.assertFalse(report['evaluation_features_read'])
            self.assertFalse(report['new_blind_test'])
            self.assertFalse(read(root/'model.json')['eligible_for_production'])


if __name__ == '__main__':
    unittest.main()
