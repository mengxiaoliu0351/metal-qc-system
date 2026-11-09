import unittest
from experiments.screw_entry_regression import summarize_rows


def row(truth, prediction=None, bound=False, view=None, expected=None, wrong=False):
    return {'sample': {'cohort':'fixture','truth':truth,'expected_view':expected},'status':'completed',
            'wall_seconds':1.,'valid_quality_prediction':bound and not wrong and truth is not None,
            'misapplied_quality_model':wrong,'result':{'perception_prediction':prediction,
            'family_routing':{'view_key':view},'disposition':{'requires_review':not bound},'vlm_called':False}}


class EntryRegressionMetricsTests(unittest.TestCase):
    def test_abstention_is_not_a_correct_normal_or_a_false_negative(self):
        values=[row('normal','normal',True,'known','known'),row('defect')]
        m=summarize_rows(values)['fixture']
        self.assertEqual(m['accuracy_classified_only'],1.)
        self.assertEqual(m['correct_quality_decisions_over_labelled_all'],.5)
        self.assertEqual(m['quality_abstained'],1)
        self.assertEqual(m['false_negatives'],0)
        self.assertEqual(m['defects_abstained'],1)

    def test_wrong_view_prediction_does_not_gain_credit_from_matching_binary_truth(self):
        m=summarize_rows([row('normal','normal',True,'wrong','expected',True)])['fixture']
        self.assertEqual(m['misapplied_quality_models'],1)
        self.assertEqual(m['valid_classified'],0)
        self.assertEqual(m['correct_quality_decisions_over_labelled_all'],0)

    def test_unlabelled_appearance_cannot_produce_quality_accuracy(self):
        m=summarize_rows([row(None)])['fixture']
        self.assertIsNone(m['accuracy_classified_only'])
        self.assertIsNone(m['correct_quality_decisions_over_labelled_all'])
        self.assertEqual(m['quality_labelled'],0)


if __name__=='__main__':unittest.main()
