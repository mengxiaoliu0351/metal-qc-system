import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perception.family import FamilyModel, conditional_features


class FamilyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        params = {"bank": np.eye(2,dtype=np.float32),"bank_group":np.array([0,1]),"centroids":np.eye(2)}
        for prefix,dim in (("pooled",2),("structured",4),("separate_0",2),("separate_1",2)):
            params.update({prefix+"_mean":np.zeros(dim),prefix+"_scale":np.ones(dim),
                           prefix+"_coef":np.ones((1,dim)),prefix+"_intercept":np.array([-1.])})
        np.savez(self.folder/"family.npz",**params)
        self.metadata={"groups":[{"key":"a","name":"View A"},{"key":"b","name":"View B"}],
            "arrays":"family.npz","arrays_sha256":hashlib.sha256((self.folder/"family.npz").read_bytes()).hexdigest(),
            "scope_min_similarity":[.9,.9],"thresholds":{"separate":[.5,.5],"pooled":.6,"structured":[.5,.9]}}
        (self.folder/"family.json").write_text(json.dumps(self.metadata))
        self.model=FamilyModel(self.folder/"family.json")

    def tearDown(self): self.temp.cleanup()

    def test_shared_bank_matches_any_group_but_tagged_subsets_remain_separate(self):
        result=self.model.patch_scores(np.array([[1.,0.],[0.,1.]]))
        np.testing.assert_array_equal(result['pooled'],[0,0])
        np.testing.assert_array_equal(result['grouped'],[[0,1],[1,0]])

    def test_routing_is_input_based_and_known_group_only_affects_oracle_comparator(self):
        a=self.model.predict([1.,0.],known_group=0);b=self.model.predict([1.,0.],known_group=1)
        self.assertEqual(a['routing']['group'],0)
        self.assertEqual(a['scores']['structured'],b['scores']['structured'])
        self.assertEqual(a['scores']['pooled'],b['scores']['pooled'])
        self.assertEqual(a['scores']['routed_expert'],b['scores']['routed_expert'])
        no_truth=self.model.predict([0.,1.])
        self.assertNotIn('separate',no_truth['scores'])
        self.assertEqual(no_truth['routing']['group'],1)
        self.assertEqual(no_truth['thresholds']['structured'],.9)

    def test_unknown_appearance_is_marked_outside_scope_and_never_authorizes_archival(self):
        result=self.model.predict([1.,1.])
        self.assertFalse(result['routing']['in_scope'])
        self.assertTrue(result['requires_review']);self.assertFalse(result['automatic_archival'])
        with self.assertRaises(ValueError):self.model.predict([np.nan,0])
        with self.assertRaises(ValueError):self.model.predict([1,0],known_group=True)

    def test_tampered_model_is_rejected(self):
        with (self.folder/'family.npz').open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):FamilyModel(self.folder/'family.json')

    def test_conditional_features_do_not_change_the_original_backbone_vector(self):
        x=np.array([[.1,.2],[.3,.4]])
        actual=conditional_features(x,[1,0],2)
        np.testing.assert_array_equal(actual[:,:2],x)
        np.testing.assert_array_equal(actual[:,2:],[[0,1],[1,0]])
        with self.assertRaises(ValueError):conditional_features(x,[2,0],2)

    def test_website_trial_uses_shared_inference_without_creating_history_or_requiring_source_label(self):
        from fastapi.testclient import TestClient
        from web import server, family_api
        detector=Mock()
        detector.extract_features.return_value=(np.tile(np.eye(2,dtype=np.float32),(2,1)),np.array([1.,0.]))
        detector._overlay.return_value=Image.new('RGB',(128,128),'gray')
        image=Image.new('RGB',(128,128),'white');image.paste('black',(0,0,64,128))
        content=io.BytesIO();image.save(content,format='PNG')
        with patch.object(family_api,'get_model',return_value=self.model),patch.object(server,'get_engine',return_value=(detector,{})),patch.object(server,'RESULTS_DIR',str(self.folder)):
            response=TestClient(server.app).post('/api/product-family/detect',files={'file':('unlabelled.png',content.getvalue(),'image/png')})
        self.assertEqual(response.status_code,200,response.text)
        result=response.json();self.assertEqual(result['routing']['reference_group'],'a')
        self.assertNotIn('separate',result['scores']);self.assertFalse(result['automatic_archival'])
        self.assertTrue(result['requires_review']);self.assertTrue(result['heatmap'])
        self.assertFalse((self.folder/'history').exists())

    def test_website_rejects_blank_image_before_loading_a_model(self):
        from fastapi.testclient import TestClient
        from web import server,family_api
        image=io.BytesIO();Image.new('RGB',(128,128),'black').save(image,format='PNG')
        with patch.object(family_api,'get_model') as get:
            response=TestClient(server.app).post('/api/product-family/detect',files={'file':('blank.png',image.getvalue(),'image/png')})
            get.assert_not_called()
        self.assertEqual(response.status_code,400)

    def test_website_reports_pending_experiment_without_affecting_catalog(self):
        from fastapi.testclient import TestClient
        from web import server
        with patch.object(server,'RESULTS_DIR',str(self.folder)):
            response=TestClient(server.app).get('/api/product-family/benchmark')
        self.assertEqual(response.status_code,200)
        self.assertFalse(response.json()['ready'])
        (self.folder/'perception').mkdir()
        (self.folder/'perception/family_active.json').write_text(json.dumps({'directory':'../../outside'}))
        with patch.object(server,'RESULTS_DIR',str(self.folder)):
            self.assertEqual(TestClient(server.app).get('/api/product-family/benchmark').status_code,503)

    def test_augmented_fit_never_reads_original_or_new_retained_test_images(self):
        from experiments import screw_family
        rows=[]
        for group in (0,1):
            for role,labels in [('bank',[0]),('train',[0,1]),('calibration',[0,1]),('test',[0,1]),('new_test',[0,1])]:
                rows += [{'group':group,'role':role,'label':label,'path':f'{group}/{role}/{label}'} for label in labels]
        protocol={'version':'fixture','samples':rows,'training_conditions':['original','dim'],'calibration_conditions':['original','dim'],
            'source_artifacts':{'weights':'a'*64},'groups':self.metadata['groups'],'coreset_per_group':4,'scope':'fixture'}
        output=self.folder/'fit';output.mkdir();(output/'protocol.json').write_text(json.dumps(protocol))
        seen=[]
        def get(sample,condition='original'):
            self.assertNotIn(sample['role'],('test','new_test'))
            seen.append((sample['role'],condition))
            vector=np.array([1.,.1]) if sample['label']==0 else np.array([.1,1.])
            return np.tile(vector/np.linalg.norm(vector),(4,1)).astype(np.float32),vector
        features=Mock();features.get.side_effect=get
        with patch.object(screw_family,'Features',return_value=features):screw_family.fit(output)
        m=json.loads((output/'family.json').read_text())
        self.assertEqual(m['training_images'],6);self.assertEqual(m['training_views'],12)
        self.assertEqual(m['calibration_images'],4);self.assertEqual(m['calibration_views'],8)
        self.assertTrue(any(c=='dim' for _,c in seen))


if __name__=='__main__':unittest.main()
