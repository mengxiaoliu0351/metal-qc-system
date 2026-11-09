import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments import m2_body_robust as robust
from products.families import read,write
from perception.body_quality import BodyQualityModel


class RobustBodyTests(unittest.TestCase):
    def test_unused_split_is_disjoint_and_preserves_publisher_test(self):
        rows=[]
        for label in ('normal','defect'):
            for i in range(350):rows.append({'pixel_hash':label+str(i),'label':label,'role':'feedback','excluded':None,'source_path':f'{label}/{i}','source_partition':'Train'})
            rows.append({'pixel_hash':label+'old-test','label':label,'role':'test','excluded':None,'source_path':'old-test','source_partition':'Test'})
        selected=robust.split_unused(rows);test={s['pixel_hash'] for s in selected if s['role']=='test'};train={s['pixel_hash'] for s in selected if s['role']=='train'}
        self.assertEqual(len(test),400);self.assertEqual(len(train),100);self.assertFalse(test&train)
        self.assertTrue(all(s['publisher_partition']=='Train' for s in selected));self.assertEqual(selected,robust.split_unused(rows))

    def test_photometric_variants_preserve_geometry(self):
        image=Image.new('RGB',(22,33),(100,120,140));dim=robust.augment(image,'dim');bright=robust.augment(image,'bright')
        self.assertEqual(dim.size,image.size);self.assertEqual(bright.size,image.size)
        self.assertLess(np.asarray(dim).mean(),np.asarray(image).mean());self.assertGreater(np.asarray(bright).mean(),np.asarray(image).mean())
        with self.assertRaises(ValueError):robust.augment(image,'warp')

    def test_threshold_minimizes_false_positive_under_recall_target(self):
        labels=np.array([0]*20+[1]*20);scores=np.r_[np.linspace(.01,.3,20),np.linspace(.4,.9,20)]
        low,high,feasible=robust.conservative_threshold(labels,scores,{'minimum_recall':.93,'maximum_false_positive_rate':.08})
        self.assertTrue(feasible);self.assertGreater(high,low);self.assertTrue(.3<low<.4)
        impossible=np.array([.5]*40);_,_,feasible=robust.conservative_threshold(labels,impossible,{'minimum_recall':.93,'maximum_false_positive_rate':.08})
        self.assertFalse(feasible)

    def test_fit_never_reads_qualification_or_test_features(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);prior=root/'prior';prior.mkdir();output=root/'v2';(output/'features').mkdir(parents=True)
            np.savez_compressed(prior/'quality.npz',bank=np.eye(2),body_centroid=[1.,0.],head_centroid=[0.,1.])
            write(prior/'quality.json',{'family_key':'screw','view':'m2_body','version':'fixture-v1','arrays':'quality.npz',
                'arrays_sha256':hashlib.sha256((prior/'quality.npz').read_bytes()).hexdigest(),'automatic_archival':False,'quality_label_basis':'publisher_OK_NG',
                'view_gate':{'min_body_similarity':.8,'margin':0},'weights_sha256':'fixture'})
            samples=[]
            for role,count in [('train',40),('calibration',20),('qualification',2),('test',2),('regression',2)]:
                for i in range(count):
                    s={'role':role,'pixel_hash':f'{role}_{i}','label':'defect' if i%2 else 'normal'};samples.append(s)
                    if role not in ('train','calibration'):continue
                    for variant in robust.VARIANTS if role=='train' else ('original',):
                        np.savez_compressed(robust.feature_path(output,s,variant),vector=[1.,.5 if i%2 else .01],raw=np.array(.1))
            write(output/'protocol.json',{'version':'fixture-v2','samples':samples,'classifier_C':[.01,.1],'policies':['original','photometric'],
                'seed':1,'calibration_target':{'minimum_recall':.93,'maximum_false_positive_rate':.08}})
            with patch.object(robust,'PRIOR',prior):robust.fit(output)
            report=read(output/'development_report.json');self.assertFalse(report['test_read']);self.assertFalse(report['qualification_read']);self.assertFalse(report['regression_read'])
            model=BodyQualityModel(output/'quality.json');self.assertFalse(model.meta['qualified_for_research_trial']);self.assertFalse(model.meta['automatic_archival'])
            self.assertEqual(model.meta['training']['source_images'],40)
            with patch.object(robust,'PRIOR',prior),self.assertRaisesRegex(ValueError,'refit'):robust.fit(output)

    def test_historical_failure_vetoes_good_fresh_metrics(self):
        gate={'minimum_recall':.9,'maximum_false_positive_rate':.1,'minimum_scope_coverage':.95}
        self.assertFalse(robust.failed_gate({'recall':.95,'false_positive_rate':.05},.99,gate))
        self.assertTrue(robust.failed_gate({'recall':.95,'false_positive_rate':.146},.99,gate))
        self.assertTrue(robust.failed_gate({'recall':.89,'false_positive_rate':.01},.99,gate))


if __name__=='__main__':unittest.main()
