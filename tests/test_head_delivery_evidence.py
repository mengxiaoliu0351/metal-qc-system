import unittest
from experiments.build_screw_delivery import head_quality_cohorts


def previous():
    return {'recipe':{'config':{'threshold':.3,'t_high':.9},'files':{'old.npz':'old-hash'}},
            'evaluation':{'normal':257,'defect':279,'local_false_positives':17,'local_false_negatives':28,'scored':536}}


class HeadDeliveryEvidenceTests(unittest.TestCase):
    def test_old_and_current_results_retain_their_own_model_and_thresholds(self):
        active={'threshold':.4,'t_high':.7};candidate={'arrays_sha256':'new-hash','thresholds':[.4,.7]}
        report={'cohorts':{'test':{'candidate':{'n':536,'confusion_matrix':[[253,4],[25,254]]}}}}
        old,new=head_quality_cohorts(previous(),'old-hash',active,'new-hash',candidate,report)
        self.assertIn('历史对照',old['name']);self.assertEqual(old['runtime_model_arrays_sha256'],'old-hash')
        self.assertEqual(old['runtime_threshold'],.3);self.assertEqual(old['metrics']['false_positives'],17)
        self.assertEqual(new['runtime_model_arrays_sha256'],'new-hash');self.assertEqual(new['runtime_threshold'],.4)
        self.assertEqual(new['metrics']['false_positives'],4);self.assertEqual(new['metrics']['false_negatives'],25)
        self.assertIn('强制人工复核',new['state']);self.assertIn('离线',new['scope'])

    def test_unmatched_current_threshold_or_model_cannot_borrow_a_report(self):
        active={'threshold':.4,'t_high':.7};report={'cohorts':{'test':{'candidate':{'confusion_matrix':[[253,4],[25,254]]}}}}
        for candidate in (None,{'arrays_sha256':'old-hash','thresholds':[.4,.7]},
                          {'arrays_sha256':'new-hash','thresholds':[.5,.7]}):
            with self.assertRaises(ValueError):head_quality_cohorts(previous(),'old-hash',active,'new-hash',candidate,report)

    def test_original_active_model_remains_supported_without_expansion_files(self):
        rows=head_quality_cohorts(previous(),'old-hash',{'threshold':.3,'t_high':.9},'old-hash')
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['id'],'m2_head')
        self.assertAlmostEqual(rows[0]['metrics']['accuracy'],491/536)
