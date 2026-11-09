import tempfile
from pathlib import Path
import unittest

from experiments.csem_detail_probe import development_rows, extract, patches


class CsemDetailIsolationTests(unittest.TestCase):
    def test_development_excludes_source_test_and_fit_defects(self):
        rows = [{'specimen_id': 'normal-fit', 'role': 'fit', 'label': 'normal'},
                {'specimen_id': 'defect-fit', 'role': 'fit', 'label': 'defect'},
                {'specimen_id': 'cal', 'role': 'calibration', 'label': 'defect'},
                {'specimen_id': 'sealed', 'role': 'test', 'label': 'normal'}]
        self.assertEqual(development_rows(rows), [rows[0], rows[2]])

    def test_same_physical_part_cannot_cross_roles_via_illumination(self):
        with self.assertRaises(ValueError):
            development_rows([{'specimen_id': 'same', 'role': 'fit', 'label': 'normal'},
                              {'specimen_id': 'same', 'role': 'test', 'label': 'normal'}])

    def test_test_reads_are_rejected_before_files_or_encoder_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for role, label in [('test', 'normal'), ('fit', 'defect')]:
                with self.assertRaises(ValueError):
                    extract(root, [{'role': role, 'label': label}])
            with self.assertRaises(ValueError):
                patches(root, {'role': 'test'})
            self.assertEqual(list(root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
