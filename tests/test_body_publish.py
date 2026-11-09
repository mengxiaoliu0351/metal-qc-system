import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'deploy/publish_body_quality.py'
spec=importlib.util.spec_from_file_location('body_publisher',path);publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)


class BodyPublicationTests(unittest.TestCase):
    def fixture(self):
        limits={'minimum_recall':.9,'maximum_false_positive_rate':.1,'minimum_scope_coverage':.95}
        protocol={'sealed_trial_gate':{**limits,'minimum_head_rejection':.95},'qualification_gate':limits.copy(),'historical_regression_gate':limits.copy()}
        metrics={'recall':.95,'false_positive_rate':.05}
        meta={'selected_mode':'global_binary','qualified_for_research_trial':True}
        report={'methods':{'global_binary':metrics.copy()},'scope_coverage':.99,'head_controls':{'n':100,'incorrectly_accepted_as_body':0},
            'qualification':{'metrics':metrics.copy(),'scope_coverage':.99},'historical_regression':{'current':metrics.copy(),'scope_coverage':.99},
            'eligible_for_research_trial':True,'trial_gate_reasons':[]}
        return meta,report,protocol

    def test_fresh_success_cannot_override_historical_failure(self):
        meta,r,p=self.fixture();r['historical_regression']['current']['false_positive_rate']=.146
        with self.assertRaisesRegex(ValueError,'permission'):publisher.validate_trial_gate(meta,r,p)
        r.update(eligible_for_research_trial=False,trial_gate_reasons=['historical failure'])
        self.assertFalse(publisher.validate_trial_gate(meta,r,p))

    def test_qualification_and_scope_failure_cannot_be_overridden(self):
        meta,r,p=self.fixture();r['qualification']['metrics']['recall']=.89
        with self.assertRaises(ValueError):publisher.validate_trial_gate(meta,r,p)
        meta,r,p=self.fixture();r['scope_coverage']=.9
        with self.assertRaises(ValueError):publisher.validate_trial_gate(meta,r,p)

    def test_gate_floors_cannot_be_relaxed_to_make_trial_pass(self):
        meta,r,p=self.fixture();p['sealed_trial_gate']['maximum_false_positive_rate']=.15
        with self.assertRaisesRegex(ValueError,'relax'):publisher.validate_trial_gate(meta,r,p)

    def test_passing_all_cohorts_and_legacy_report_are_supported(self):
        meta,r,p=self.fixture();self.assertTrue(publisher.validate_trial_gate(meta,r,p))
        del p['historical_regression_gate'];del r['historical_regression'];del r['qualification']
        self.assertTrue(publisher.validate_trial_gate(meta,r,p))


if __name__=='__main__':unittest.main()
