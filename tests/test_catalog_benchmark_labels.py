import unittest
import numpy as np
from review.workflow import benchmark_labels


class CatalogBenchmarkLabelTests(unittest.TestCase):
    def test_imported_normal_folder_is_not_treated_as_defective(self):
        paths=['images/train/normal/a.png','images/train/defect/b.png']
        np.testing.assert_array_equal(benchmark_labels({'labels':np.array([0,1])},paths,True),[0,1])

    def test_missing_or_invalid_import_labels_fail_instead_of_guessing(self):
        for data in ({},{'labels':[0]},{'labels':[0,2]},{'labels':[0,float('nan')]}):
            with self.assertRaises(ValueError):benchmark_labels(data,['a','b'],True)

    def test_legacy_mvtec_folder_rule_is_preserved(self):
        np.testing.assert_array_equal(benchmark_labels({},['train/good/a.png','test/bent/b.png']),[0,1])
