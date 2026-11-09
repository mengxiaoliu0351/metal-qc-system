import hashlib
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from perception.appearance_boundary import BoundaryAppearanceMatcher,load_matcher
from products.families import FamilyRegistry,write
from experiments.screw_appearance_boundary import promotion_report


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        np.savez_compressed(self.root/'appearance.npz',clip=np.eye(2),labels=np.array([5,6]),exclusion_clip=np.array([[1.,0.]]),exclusion_labels=np.array([4]))
        self.meta={'arrays':'appearance.npz','arrays_sha256':hashlib.sha256((self.root/'appearance.npz').read_bytes()).hexdigest(),
            'matcher_kind':'source-exclusion-v1','selected_method':'clip','categories':[{'category_id':5,'name':'亮面'},{'category_id':6,'name':'黑色'}],
            'reference_keys':['a','b'],'exclusion_keys':['c'],'gate':{'similarity':.9,'margin':0.,'exclusion_margin':.01},
            'quality_labels_used':False,'automatic_archival':False,'version':'fixture'}
        write(self.root/'appearance.json',self.meta)

    def tearDown(self):self.temp.cleanup()

    def test_close_exclusion_abstains_without_calling_it_a_defect(self):
        model=load_matcher(self.root/'appearance.json');value=model.predict([1.,0.])
        self.assertIsNone(value['category_id']);self.assertFalse(value['matched']);self.assertEqual(value['exclusion_category_id'],4)
        self.assertEqual(value['quality_assessment'],'not_available');self.assertFalse(value['automatic_archival'])
        self.assertTrue(model.predict([0.,1.])['matched'])

    def test_exclusion_cannot_share_a_supported_category(self):
        np.savez_compressed(self.root/'appearance.npz',clip=np.eye(2),labels=np.array([5,6]),exclusion_clip=np.array([[1.,0.]]),exclusion_labels=np.array([5]))
        self.meta['arrays_sha256']=hashlib.sha256((self.root/'appearance.npz').read_bytes()).hexdigest();write(self.root/'appearance.json',self.meta)
        with self.assertRaisesRegex(ValueError,'overlap'):BoundaryAppearanceMatcher(self.root/'appearance.json')

    def test_conservative_metadata_cannot_relax_the_original_gate(self):
        self.meta['fixed_positive_gate']={'similarity':.95,'margin':0.}
        write(self.root/'appearance.json',self.meta)
        with self.assertRaisesRegex(ValueError,'cannot widen'):BoundaryAppearanceMatcher(self.root/'appearance.json')

    def test_rejecting_everything_cannot_pass_promotion(self):
        baseline={'known_correct_acceptance':.9,'known_coverage':.9,'other_rejection':.9,'accepted_precision_all':.9,'accepted_known_wrong':0}
        improved={**baseline,'other_rejection':.98,'accepted_precision_all':.97}
        measurements={part:{'v1':baseline.copy(),'v2':improved.copy()} for part in ('fresh_test','regression')}
        measurements['control']={'v1':{'accepted_other':0},'v2':{'accepted_other':0}}
        rules={'max_known_correct_acceptance_drop':.02,'min_fresh_known_coverage':.8}
        self.assertTrue(promotion_report(measurements,rules)['eligible'])
        measurements['fresh_test']['v2'].update(known_correct_acceptance=0.,known_coverage=0.,other_rejection=1.)
        self.assertFalse(promotion_report(measurements,rules)['eligible'])

    def test_reserved_train_image_cannot_enter_a_future_reference_build(self):
        from experiments import screw_appearance
        write(self.root/'results/product_families/screw/appearance.json',{'summary':{'samples':1,'types':1},'types':[],'samples':[{'key':'reserved','split':'train','category_id':5,'source_frame':'frame'}]})
        write(self.root/'results/product_families/screw/appearance_reservations.json',{'samples':{'reserved':{'source_frame':'frame'}}})
        write(self.root/'out/review_snapshot.json',{'reviews':[{'key':'reserved','appearance_status':'accepted'}]})
        with patch.object(screw_appearance,'ROOT',self.root):
            with self.assertRaisesRegex(ValueError,'Frozen evaluation'):screw_appearance.prepare(self.root/'out')
        registry=FamilyRegistry(self.root/'results');sample=registry.samples('screw')['samples'][0]
        self.assertTrue(sample['evaluation_reserved']);self.assertFalse(sample['eligible_normal_reference'])
