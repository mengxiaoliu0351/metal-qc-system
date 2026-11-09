import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image,ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from perception.appearance import AppearanceMatcher,color_shape,full_object,ranked
from products.families import write


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.folder=self.root/'experiments/screw_appearance_v1';self.folder.mkdir(parents=True)
        self.image=Image.new('RGB',(50,160),'tan');ImageDraw.Draw(self.image).rectangle((20,10,30,150),fill='black')
        other=Image.new('RGB',(50,160),'tan');ImageDraw.Draw(other).rectangle((15,20,35,140),fill='white')
        self.references=np.array([color_shape(self.image),color_shape(other)])
        self.keys=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(2)]
        np.savez_compressed(self.folder/'appearance.npz',color_shape=self.references,labels=np.array([5,6]))
        self.meta={'version':'fixture','family_key':'screw','arrays':'appearance.npz','arrays_sha256':hashlib.sha256((self.folder/'appearance.npz').read_bytes()).hexdigest(),
            'protocol_sha256':'fixture','categories':[{'category_id':5,'name':'亮面'},{'category_id':6,'name':'黑色'}],'reference_keys':self.keys,
            'selected_method':'color_shape','gate':{'similarity':.9,'margin':.005},'quality_labels_used':False,'automatic_archival':False}
        write(self.folder/'appearance.json',self.meta);write(self.folder/'report.json',{'arrays_sha256':self.meta['arrays_sha256'],'protocol_sha256':'fixture'})
        write(self.root/'product_families/screw/appearance_active.json',{'directory':'experiments/screw_appearance_v1'})
        self.model=AppearanceMatcher(self.folder/'appearance.json')

    def tearDown(self):self.temp.cleanup()

    def test_padding_preserves_both_ends_of_narrow_object(self):
        image=Image.new('RGB',(20,200),'white');d=ImageDraw.Draw(image);d.rectangle((0,0,19,12),fill='red');d.rectangle((0,185,19,199),fill='blue')
        x=np.asarray(full_object(image));self.assertEqual(x.shape,(224,224,3))
        self.assertTrue(((x[:,:,0]>200)&(x[:,:,1]<50)).any());self.assertTrue(((x[:,:,2]>200)&(x[:,:,0]<50)).any())

    def test_matching_never_produces_quality_truth(self):
        value=self.model.predict(self.references[0]);self.assertTrue(value['matched']);self.assertEqual(value['category_id'],5)
        self.assertEqual(value['quality_assessment'],'not_available');self.assertFalse(value['automatic_archival'])
        self.assertNotIn('defect',value);self.assertNotIn('normal',value)

    def test_ambiguous_reference_scores_abstain(self):
        pred=ranked(self.references[0],np.stack([self.references[0],self.references[0]]),[5,6],[5,6])
        self.assertEqual(float(pred['margin'][0]),0.)
        self.model.meta['gate']['similarity']=1.01
        self.assertIsNone(self.model.predict(self.references[0])['category_id'])

    def test_tampered_reference_or_quality_permission_rejected(self):
        meta=dict(self.meta);meta['automatic_archival']=True;write(self.folder/'appearance.json',meta)
        with self.assertRaises(ValueError):AppearanceMatcher(self.folder/'appearance.json')
        write(self.folder/'appearance.json',self.meta)
        with (self.folder/'appearance.npz').open('ab') as f:f.write(b'changed')
        with self.assertRaises(ValueError):AppearanceMatcher(self.folder/'appearance.json')

    def test_http_appearance_trial_does_not_create_detection_history(self):
        from fastapi.testclient import TestClient
        from web import server
        content=io.BytesIO();self.image.save(content,format='PNG')
        with patch.object(server,'RESULTS_DIR',str(self.root)):
            client=TestClient(server.app)
            self.assertTrue(client.get('/api/product-families/screw/matching').json()['ready'])
            response=client.post('/api/product-families/screw/matching/detect',files={'file':('part.png',content.getvalue(),'image/png')})
        self.assertEqual(response.status_code,200,response.text);value=response.json()
        self.assertEqual(value['quality_assessment'],'not_available');self.assertFalse(value['automatic_archival'])
        self.assertEqual(len(value['references']),2);self.assertFalse((self.root/'history').exists())

    def test_blank_image_rejected_before_model_or_encoder(self):
        from fastapi.testclient import TestClient
        from web import server,family_api
        content=io.BytesIO();Image.new('RGB',(50,160),'black').save(content,format='PNG')
        with patch.object(server,'RESULTS_DIR',str(self.root)),patch.object(family_api,'get_appearance_model') as load:
            response=TestClient(server.app).post('/api/product-families/screw/matching/detect',files={'file':('blank.png',content.getvalue(),'image/png')})
            self.assertEqual(response.status_code,400);load.assert_not_called()

    def test_unbuilt_or_mismatched_family_cannot_fall_back_to_screw_library(self):
        from fastapi.testclient import TestClient
        from web import server
        with patch.object(server,'RESULTS_DIR',str(self.root)):
            client=TestClient(server.app);key=client.post('/api/product-families',json={'name':'螺母族'}).json()['family']['key']
            self.assertFalse(client.get(f'/api/product-families/{key}/matching').json()['ready'])
            write(self.root/f'product_families/{key}/appearance_active.json',{'directory':'experiments/screw_appearance_v1'})
            self.assertEqual(client.get(f'/api/product-families/{key}/matching').status_code,503)

    def test_report_includes_unknown_false_accepts_in_precision(self):
        from experiments.screw_appearance import stats
        rows=[{'category_id':5},{'category_id':4},{'category_id':6}]
        pred={'category':np.array([5,5,5]),'similarity':np.array([1.,1.,1.]),'margin':np.array([.1,.1,.1])}
        result=stats(rows,pred,[5,6],{'similarity':.9,'margin':0})
        self.assertEqual(result['accepted_other'],1);self.assertEqual(result['accepted_known_wrong'],1)
        self.assertAlmostEqual(result['accepted_precision_all'],1/3)

    def test_preparation_rejects_a_test_image_approved_as_reference(self):
        from experiments import screw_appearance
        manifest=self.root/'results/product_families/screw/appearance.json'
        write(manifest,{'samples':[{'key':'heldout','split':'test','category_id':5}]})
        output=self.root/'experiment';write(output/'review_snapshot.json',{'reviews':[{'key':'heldout','appearance_status':'accepted'}]})
        with patch.object(screw_appearance,'ROOT',self.root):
            with self.assertRaisesRegex(ValueError,'Only reviewed train'):screw_appearance.prepare(output)


if __name__=='__main__':unittest.main()
