import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from interface import report
from products.families import read,write
from review.inspection_context import INSPECTION_CONTEXT
from review.workflow import ReviewWorkflow
from web import server,family_inspection as api


class UnifiedFamilyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.results=self.root/'results';self.results.mkdir()
        self.workflow=ReviewWorkflow(self.root,self.results);self.policy=Mock();self.workflow._inspection_policy=self.policy
        def decide(record,config,vector):
            if config and config['key']=='screw':return {'status':'auto_normal','label':'自动归档','requires_review':False,'auto_result':'normal'}
            return {'status':'needs_review','label':'待人工复核','requires_review':True,'auto_result':None,'reason':'已有门槛未通过或尚未绑定模型'}
        self.policy.decide.side_effect=decide;self.policy.observe_review.return_value=False
        folder=self.root/'router';folder.mkdir();write(folder/'report.json',{'operational_routing_ready':True})
        self.route={'matched':True,'view_key':'mvtec_screw','view_name':'螺丝近景视图','reason':'匹配参考','router_version':'fixture-router'}
        self.model=SimpleNamespace(path=folder/'router.json',predict=Mock(side_effect=lambda _:dict(self.route)))
        self.detector=Mock();self.detector.extract_features.return_value=(np.ones((196,3)),np.ones(3))
        appearance=Mock();appearance.predict.return_value={'matched':False}
        self.patchers=[patch.object(server,'ROOT',str(self.root)),patch.object(server,'RESULTS_DIR',str(self.results)),
            patch.object(report,'HISTORY_DIR',str(self.results/'history')),patch.object(server,'get_review_workflow',return_value=self.workflow),
            patch.object(api,'load_router',return_value=self.model),patch.object(server,'get_engine',return_value=(self.detector,{})),
            patch('web.family_api.get_appearance_model',return_value=appearance),
            patch.object(server,'load_products',return_value=[{'key':'screw'},{'key':'product_a269405b7b9a'}]),
            patch.object(server,'_locked_detection',side_effect=self.production)]
        for p in self.patchers:p.start()
        self.production_mock=server._locked_detection;self.appearance=appearance;self.client=TestClient(server.app)
        image=Image.fromarray(np.tile(np.arange(200,dtype=np.uint8),(200,1))).convert('RGB');buffer=io.BytesIO();image.save(buffer,format='PNG');self.content=buffer.getvalue()

    def tearDown(self):
        for p in reversed(self.patchers):p.stop()
        self.workflow.executor.shutdown(wait=True);self.temp.cleanup()

    def production(self,content,filename,product):
        image=Image.open(io.BytesIO(content)).convert('RGB')
        return server._finalize_detection(image,filename,{'key':product,'name':product},{'heatmap':image,'image_features':np.ones(3)},time.perf_counter(),
            {'ok':True,'route':'low_risk','score':.1,'threshold':.3,'t_high':.7,'vlm_called':False,'heatmap':'fixture',
             'perception_prediction':'normal','cognition':{'conclusion':'正常线索'}})

    def upload(self,content=None):return self.client.post('/api/product-families/screw/inspection/detect',files={'file':('part.png',content or self.content,'image/png')})

    def test_known_view_uses_existing_auto_policy_and_persists_routing(self):
        response=self.upload();self.assertEqual(response.status_code,200,response.text);d=response.json()
        self.assertEqual(self.production_mock.call_args.args[2],'screw');self.assertEqual(d['disposition']['status'],'auto_normal')
        _,record=self.workflow.locate(session_id=d['session_id']);self.assertEqual(record['family_key'],'screw');self.assertEqual(record['family_routing']['view_key'],'mvtec_screw')
        self.assertEqual(record['product_key'],'screw');self.assertEqual(INSPECTION_CONTEXT.get(),{})
        self.assertEqual(self.client.get('/api/product-families/screw/inspection/metrics').json()['summary']['new_policy_records'],1)

    def test_existing_head_hold_is_not_cleared(self):
        self.route.update(view_key='m2_head',view_name='M2头部俯视');d=self.upload().json()
        self.assertEqual(self.production_mock.call_args.args[2],'product_a269405b7b9a');self.assertTrue(d['disposition']['requires_review']);self.assertIsNone(d['disposition']['auto_result'])

    def test_body_candidate_does_not_bypass_gate_and_can_be_reviewed_without_training(self):
        self.route.update(view_key='m2_body',view_name='M2螺杆侧视');response=self.upload();self.assertEqual(response.status_code,200,response.text);d=response.json()
        self.production_mock.assert_not_called();self.assertIsNone(d['score']);self.assertTrue(d['disposition']['requires_review']);self.assertFalse(d['iteration_eligible'])
        with self.assertRaisesRegex(ValueError,'尚未绑定'):self.workflow.confirm({'session_id':d['session_id'],'verdict':'ok','reviewer':'fixture','use_for_iteration':True})
        reviewed=self.workflow.confirm({'session_id':d['session_id'],'verdict':'ok','reviewer':'fixture','use_for_iteration':False})
        self.assertTrue(reviewed['ok']);self.assertIsNone(reviewed['annotation']['agreement']);self.assertFalse(reviewed['learned'])
        pdf=self.client.get('/api/report/'+d['session_id']);self.assertEqual(pdf.status_code,200);self.assertTrue(pdf.content.startswith(b'%PDF'))
        import fitz
        with fitz.open(stream=pdf.content,filetype='pdf') as document:text=''.join(p.get_text() for p in document)
        self.assertIn('未进行质量判定',text);self.assertIn('M2螺杆侧视',text)

    def test_appearance_match_cannot_authorize_quality_classification(self):
        self.appearance.predict.return_value={'matched':True,'category_name':'亮面螺丝'};d=self.upload().json()
        self.production_mock.assert_not_called();self.assertEqual(d['family_action'],'review_only');self.assertIsNone(d['score']);self.assertIsNone(d['family_routing']['view_key'])

    def test_unknown_and_bad_input_create_review_records_without_guessing_labels(self):
        self.route.update(matched=False,view_key=None,view_name='未覆盖');d=self.upload().json();self.production_mock.assert_not_called();self.assertIsNone(d['perception_prediction'])
        blank=io.BytesIO();Image.new('RGB',(200,200),'white').save(blank,format='PNG');before=self.detector.extract_features.call_count;d=self.upload(blank.getvalue()).json()
        self.assertEqual(self.detector.extract_features.call_count,before);self.assertTrue(d['disposition']['requires_review']);self.assertEqual(INSPECTION_CONTEXT.get(),{})

    def test_unqualified_router_never_dispatches_and_invalid_family_isolated(self):
        write(self.model.path.parent/'report.json',{'operational_routing_ready':False})
        self.assertEqual(self.upload().status_code,409);self.production_mock.assert_not_called()
        response=self.client.post('/api/product-families/unknown/inspection/detect',files={'file':('part.png',self.content,'image/png')})
        self.assertEqual(response.status_code,404)

    def test_family_batch_freezes_each_destination_and_recovers_routing_fields(self):
        folder=self.model.path.parent
        for name in ('router.json','router.npz','protocol.json'):(folder/name).write_text('fixture',encoding='utf8')
        appearance=self.root/'appearance';appearance.mkdir()
        for name in ('appearance.json','appearance.npz','report.json'):(appearance/name).write_text('fixture',encoding='utf8')
        self.appearance.path=appearance/'appearance.json';self.appearance.meta={'arrays':'appearance.npz'}
        write(self.results/'product_families/screw/appearance_active.json',{'directory':'fixture'})
        for name in ('src/web/family_inspection.py','src/perception/family_router.py','src/perception/appearance.py','src/review/inspection_context.py'):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture',encoding='utf8')
        banks=self.results/'banks';banks.mkdir()
        configs=[{'key':key,'name':key,'bank':key+'.npy','threshold':.3,'t_high':.7} for key in ('screw','product_a269405b7b9a')]
        for config in configs:(banks/config['bank']).write_bytes(b'fixture')
        with patch.object(server,'BANKS_DIR',str(banks)),patch.object(server,'_batch_service',None),patch.object(server,'load_products',return_value=configs):
            service=server.get_batch_service()
            try:
                job=service.create('family:screw');self.assertEqual(set(job['recipe']['destinations']),{'screw','product_a269405b7b9a'})
                service.add(job['id'],self.content,'batch.png');service.start(job['id']);service.futures[job['id']].result(10)
                detail=service.detail(job['id']);self.assertEqual(detail['status'],'completed');result=detail['items'][0]['result']
                _,record=self.workflow.locate(session_id=result['session_id']);self.assertEqual(record['batch_id'],job['id']);self.assertEqual(record['family_action'],'product_workflow')
                recovered=service.recover({'batch_id':job['id'],'batch_item_id':detail['items'][0]['id'],'batch_attempt':1,'record_scope':'production'})
                self.assertEqual(recovered['family_routing']['view_key'],'mvtec_screw');self.assertTrue(recovered['quality_model_bound'])
                before=service.recipe('family:screw');(banks/configs[1]['bank']).write_bytes(b'different-head-version')
                self.assertNotEqual(before,service.recipe('family:screw'))
            finally:service.executor.shutdown(wait=True)


if __name__=='__main__':unittest.main()
