from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments import csem_screw_pilot as pilot
from products.families import read, write


class CsemPilotTests(unittest.TestCase):
    def test_specimen_split_is_deterministic_and_test_cannot_move_into_development(self):
        rows = []
        for label in ('normal', 'defect'):
            for i in range(16): rows.append({'specimen_id': label+str(i), 'publisher_partition': 'Train', 'label': label})
            for i in range(19): rows.append({'specimen_id': label+'held'+str(i), 'publisher_partition': 'Test', 'label': label})
        roles = pilot.specimen_roles(rows)
        self.assertEqual(roles, pilot.specimen_roles(list(reversed(rows))))
        self.assertEqual(sum(role == 'test' for role in roles.values()), 38)
        self.assertEqual(sum(role == 'fit' for role in roles.values()), 18)
        for s in rows:
            if s['publisher_partition'] == 'Test': self.assertEqual(roles[s['specimen_id']], 'test')

    def test_fit_never_extracts_or_reads_test_images(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder); (output/'features').mkdir()
            samples = []
            for role in ('fit', 'calibration', 'test'):
                for label in ('normal', 'defect'):
                    for specimen in range(2):
                        for light in ('501', '507', '901'):
                            key = f'{role}_{label}_{specimen}_{light}'
                            s = {'role': role, 'label': label, 'specimen_id': f'{role}_{label}_{specimen}', 'pixel_hash': key}
                            samples.append(s)
                            if role == 'test': continue
                            vector = np.array([1., .05]) if label == 'normal' else np.array([.05, 1.])
                            vector = vector/np.linalg.norm(vector)
                            np.savez_compressed(output/'features'/(key+'.npz'), patches=np.tile(vector, (4, 1)), vector=vector)
            p = {'version': 'fixture', 'samples': samples, 'bank_size': 2, 'seed': 1, 'classifier_C': .1,
                 'calibration_target': {'minimum_recall': .85, 'maximum_false_positive_rate': .15}}
            write(output/'protocol.json', p)
            def extract(output, rows): self.assertTrue(all(s['role'] in ('fit', 'calibration') for s in rows))
            with patch.object(pilot, 'verify', return_value=p), patch.object(pilot, 'extract_rows', side_effect=extract):
                pilot.fit(output)
            self.assertFalse(read(output/'development_report.json')['test_features_read'])
            self.assertFalse(read(output/'candidate.json')['automatic_archival'])


if __name__ == '__main__': unittest.main()
