import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock

import numpy as np

from experiments.prepare_m2_body import safe_target,assign_roles
from perception.body_quality import BodyQualityModel,body_scores
from products.families import write


class BodyQualityTests(unittest.TestCase):
    def test_archive_paths_cannot_escape_or_use_drives(self):
        root=Path(tempfile.gettempdir())/'body-fixture'
        for name in ('../outside.bmp','C:/outside.bmp','/outside.bmp','BODY/../../outside.bmp'):
            with self.assertRaises(ValueError):safe_target(root,name)
        self.assertTrue(safe_target(root,'BODY/Train/OK/1.BMP').is_relative_to(root.resolve()))

    def test_roles_keep_test_and_feedback_out_of_development(self):
        rows=[]
        for label in ('normal','defect'):
            for i in range(500):rows.append({'pixel_hash':label+str(i),'label':label,'source_partition':'Train','source_path':f'{label}/{i}','excluded':None,'role':None})
            rows.append({'pixel_hash':label+'test','label':label,'source_partition':'Test','source_path':label+'/test','excluded':None,'role':None})
        rows=assign_roles(rows)
        self.assertEqual(sum(s['role']=='bank' for s in rows),80)
        self.assertEqual(sum(s['role']=='train' for s in rows),400)
        self.assertEqual(sum(s['role']=='calibration' for s in rows),200)
        self.assertEqual(sum(s['role']=='qualification' for s in rows),200)
        self.assertTrue(all(s['role']=='test' for s in rows if s['source_partition']=='Test'))

    def test_head_view_never_receives_body_quality_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);np.savez_compressed(root/'quality.npz',bank=np.eye(2),body_centroid=np.array([1.,0.]),head_centroid=np.array([0.,1.]))
            meta={'arrays':'quality.npz','arrays_sha256':hashlib.sha256((root/'quality.npz').read_bytes()).hexdigest(),
                'automatic_archival':False,'quality_label_basis':'publisher_OK_NG','version':'fixture','selected_mode':'normal_memory',
                'thresholds':{'normal_memory':[.3,.6]},'view_gate':{'min_body_similarity':.8,'margin':0.}}
            write(root/'quality.json',meta);model=BodyQualityModel(root/'quality.json')
            body=model.predict([1,0],.8);head=model.predict([0,1],.8)
            self.assertEqual(body['normal_defect_hint'],'defect');self.assertTrue(body['requires_review']);self.assertFalse(body['automatic_archival'])
            self.assertIsNone(head['normal_defect_hint']);self.assertIsNone(head['score']);self.assertEqual(head['quality_assessment'],'not_available')

    def test_scores_do_not_need_or_accept_source_truth_labels(self):
        np.testing.assert_equal(body_scores([[1,0],[0,1]],[.1,.9],{},'normal_memory'),[.1,.9])
        with self.assertRaises(ValueError):body_scores([[1,0]],[.1],{},'unsupported')

    def test_fit_uses_development_only_and_exports_portable_models(self):
        from experiments.m2_body_quality import fit
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'features';cache.mkdir();rows=[]
            for role,n in [('train',40),('calibration',20),('qualification',20),('head_reference',10),('head_calibration',20),('test',2)]:
                for i in range(n):
                    label=i%2;key=f'{role}_{i}';head=role.startswith('head')
                    rows.append({'role':role,'pixel_hash':key,'label':'defect' if label else 'normal'})
                    if role=='test':continue
                    vector=np.array([0.,1.]) if head else np.array([1.,.2 if label else .02])
                    np.savez_compressed(cache/(key+'.npz'),vector=vector,raw=np.array(.8 if label else .1))
            np.savez_compressed(root/'normal_bank.npz',bank=np.eye(2),body_centroid=np.array([1.,0.]))
            write(root/'protocol.json',{'version':'fixture','samples':rows,'classifier_C':.1,'seed':1,'fusion_oof_folds':5,
                'scope_margin_grid':[-.04,0,.04],'weights_sha256':'fixture'})
            fit(root)
            self.assertFalse(__import__('json').loads((root/'development_report.json').read_text())['test_read'])
            self.assertFalse((cache/'test_0.npz').exists())
            self.assertFalse(BodyQualityModel(root/'quality.json').meta['automatic_archival'])

    def test_unlabelled_body_crop_preserves_object_and_rejects_large_table(self):
        from PIL import Image,ImageDraw
        from perception.body_preprocess import prepare_body
        image=Image.new('RGB',(400,500),'white');draw=ImageDraw.Draw(image)
        draw.rectangle((190,180,210,280),fill='black');draw.rectangle((0,370,399,499),fill='black')
        prepared,region=prepare_body(image)
        self.assertTrue(region['proposal_found']);x1,y1,x2,y2=region['source_box']
        self.assertLessEqual(x1,190);self.assertGreaterEqual(x2,211);self.assertLessEqual(y1,180);self.assertGreaterEqual(y2,281)
        self.assertLess(y2,370);self.assertEqual(prepared.size,(224,224))

    def test_http_trial_is_blocked_until_gate_and_never_archives(self):
        import io
        from PIL import Image,ImageDraw
        from fastapi.testclient import TestClient
        from web import server
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);results=root/'results';folder=results/'experiments/body_fixture';folder.mkdir(parents=True)
            weights=root/'weights/CLIP-ViT-B-16-open_clip_model.safetensors';weights.parent.mkdir();weights.write_bytes(b'fixture')
            preprocessing=root/'src/perception/body_preprocess.py';preprocessing.parent.mkdir(parents=True);preprocessing.write_bytes(b'fixed-preprocessing')
            extractor=root/'src/perception/winclip_patch.py';extractor.write_bytes(b'fixed-extractor')
            write(folder/'protocol.json',{'preprocessing_sha256':hashlib.sha256(preprocessing.read_bytes()).hexdigest(),'extractor_sha256':hashlib.sha256(extractor.read_bytes()).hexdigest()})
            protocol_sha=hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest()
            np.savez_compressed(folder/'quality.npz',bank=np.eye(2),body_centroid=np.array([1.,0.]),head_centroid=np.array([0.,1.]))
            meta={'family_key':'screw','arrays':'quality.npz','arrays_sha256':hashlib.sha256((folder/'quality.npz').read_bytes()).hexdigest(),'protocol_sha256':protocol_sha,
                'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'automatic_archival':False,'quality_label_basis':'publisher_OK_NG',
                'version':'fixture','selected_mode':'normal_memory','thresholds':{'normal_memory':[.3,.6]},'view_gate':{'min_body_similarity':.8,'margin':0.}}
            write(folder/'quality.json',meta);report={'arrays_sha256':meta['arrays_sha256'],'protocol_sha256':protocol_sha,'eligible_for_research_trial':False};write(folder/'report.json',report)
            write(results/'product_families/screw/body_quality_candidate.json',{'directory':'experiments/body_fixture'})
            image=Image.new('RGB',(200,300),'white');ImageDraw.Draw(image).rectangle((90,80,110,200),fill='black');content=io.BytesIO();image.save(content,format='PNG')
            detector=Mock();detector.extract_features.return_value=(np.tile(np.eye(2),(98,1)),np.array([1.,0.]));detector._overlay.return_value=Image.new('RGB',(224,224),'gray')
            with patch.object(server,'ROOT',str(root)),patch.object(server,'RESULTS_DIR',str(results)),patch.object(server,'get_engine',return_value=(detector,{})) as engine:
                client=TestClient(server.app)
                self.assertFalse(client.get('/api/product-families/screw/body-quality').json()['trial_ready'])
                response=client.post('/api/product-families/screw/body-quality/detect',files={'file':('part.png',content.getvalue(),'image/png')})
                self.assertEqual(response.status_code,409);engine.assert_not_called()
                report['eligible_for_research_trial']=True;write(folder/'report.json',report)
                response=client.post('/api/product-families/screw/body-quality/detect',files={'file':('part.png',content.getvalue(),'image/png')})
                self.assertEqual(response.status_code,200,response.text);self.assertFalse(response.json()['automatic_archival']);self.assertTrue(response.json()['requires_review'])
                calls=engine.call_count;preprocessing.write_bytes(b'changed-preprocessing-must-not-silently-infer')
                self.assertEqual(client.get('/api/product-families/screw/body-quality').status_code,503)
                self.assertEqual(client.post('/api/product-families/screw/body-quality/detect',files={'file':('part.png',content.getvalue(),'image/png')}).status_code,503)
                self.assertEqual(engine.call_count,calls)
            self.assertFalse((results/'history').exists())
