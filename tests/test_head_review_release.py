import copy
from pathlib import Path
import tempfile
import unittest

from deploy.release_head_review_only import KEY,VERSION,install,rollback,read,write,sha,validate_gate


def metric():
    return {'n':200,'confusion_matrix':[[95,5],[5,95]],'accuracy':.95,
            'balanced_accuracy':.95,'recall':.95,'false_positive_rate':.05}


def evidence():
    candidate={'release_permission':False,'base_calibration':metric(),'calibration':metric()}
    report={'release_permission':False,'cohorts':{role:{'current':metric(),'candidate':metric()}
            for role in ('qualification','test','family_regression')}}
    return report,candidate


class HeadReviewReleaseTests(unittest.TestCase):
    def test_derived_metric_mismatch_and_automatic_permission_are_rejected(self):
        report,candidate=evidence();validate_gate(report,candidate)
        bad=copy.deepcopy(report);bad['cohorts']['test']['candidate']['accuracy']=1.
        with self.assertRaises(ValueError):validate_gate(bad,candidate)
        with self.assertRaises(ValueError):validate_gate(report,{**candidate,'release_permission':True})

    def test_install_and_rollback_preserve_other_products_and_prior_holds(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'root';stage=Path(folder)/'stage';backup=Path(folder)/'backup'
            bank=root/'results/perception/banks/bank.npy';bank.parent.mkdir(parents=True);bank.write_bytes(b'normal-bank')
            old={'key':KEY,'bank':'bank.npy','bank_sha256':sha(bank)};other={'key':'screw','sentinel':'keep'}
            write(bank.parent/'products.json',[other,old]);write(root/'results/inspection_policy.json',{'holds':{'old-head':{'keep':True}},'revision':'old','sample_rate':.1})
            new={**old,'perception_artifact':'candidate.json','threshold':.4,'t_high':.8}
            meta=stage/'results/perception/optimized/candidate.json'
            write(meta,{'product':KEY,'mode':'global_binary','arrays':'candidate.npz','threshold':.4,'t_high':.8,'automatic_archival':False})
            model=meta.parent/'candidate.npz';model.write_bytes(b'portable-model')
            report,candidate=evidence();write(stage/'report.json',report);write(stage/'candidate.json',candidate)
            write(stage/'release.json',{'version':VERSION,'product':KEY,'automatic_archival':False,'baseline_config':old,'config':new,
                  'files':{str(p.relative_to(stage)):sha(p) for p in (meta,model)},'report_sha256':sha(stage/'report.json'),'candidate_sha256':sha(stage/'candidate.json')})
            install(root,stage,backup)
            installed=read(bank.parent/'products.json');policy=read(root/'results/inspection_policy.json')
            self.assertEqual(installed,[other,new]);self.assertEqual(policy['holds']['old-head'],{'keep':True})
            self.assertEqual(len(policy['holds']),2);self.assertEqual(policy['sample_rate'],.1)
            receipt=read(backup/'receipt.json');self.assertTrue(policy['holds'][receipt['hold_key']]['automatic_release_vetoed'])
            rollback(root,backup)
            self.assertEqual(read(bank.parent/'products.json'),[other,old])
            self.assertEqual(len(read(root/'results/inspection_policy.json')['holds']),2)

    def test_missing_cohort_cannot_be_published(self):
        report,candidate=evidence();report['cohorts'].pop('family_regression')
        with self.assertRaises(ValueError):validate_gate(report,candidate)
