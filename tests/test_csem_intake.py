import unittest
import numpy as np

from experiments.prepare_csem_screw import safe_member, specimen_label


class CsemIntakeTests(unittest.TestCase):
    def test_unannotated_numeric_specimen_is_never_normal(self):
        self.assertIsNone(specimen_label('Unannotated', '421', None))
        self.assertIsNone(specimen_label('Unannotated', 'defect_001', np.ones((2, 2))))

    def test_annotated_conventions_require_mask_consistency(self):
        self.assertEqual(specimen_label('Train', '001', np.zeros((2, 2))), 'normal')
        self.assertEqual(specimen_label('Test', 'defect_001', np.ones((2, 2))), 'defect')
        with self.assertRaises(ValueError): specimen_label('Train', 'defect_001', None)
        self.assertEqual(specimen_label('Test', '198', np.ones((2, 2))), 'defect')
        with self.assertRaises(ValueError): specimen_label('Train', '001', None)
        with self.assertRaises(ValueError): specimen_label('Train', 'unlabelled_name', None)

    def test_member_paths_cannot_escape_destination(self):
        for name in ('../escape.png', '/absolute.png', 'C:/escape.png', 'Screw\\Train\\x.png'):
            with self.assertRaises(ValueError): safe_member(name)


if __name__ == '__main__': unittest.main()
