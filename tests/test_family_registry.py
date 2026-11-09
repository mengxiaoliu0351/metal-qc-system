import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from products.families import FamilyRegistry,write


class FamilyRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.registry=FamilyRegistry(self.root)
        samples=[]
        for index,split in enumerate(('train','validation','test')):
            key=hashlib.sha256(str(index).encode()).hexdigest();path=f'product_families/screw/crops/{key}.jpg'
            target=self.root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b'image')
            samples.append({'key':key,'path':path,'split':split,'category_id':1,'category_name':'螺丝外观'})
        self.samples=samples
        write(self.root/'product_families/screw/appearance.json',{'samples':samples,'types':[{'category_id':1,'name':'螺丝外观'}],'summary':{'types':1,'samples':3}})

    def tearDown(self):self.temp.cleanup()

    def test_new_family_does_not_reuse_screw_model_or_candidates(self):
        family=self.registry.create({'name':'螺母族'})
        self.assertFalse(self.registry.list()[1]['model_ready'])
        self.assertEqual(self.registry.samples(family['key'])['count'],0)
        with self.assertRaises(FileNotFoundError):self.registry.artifact(family['key'])
        with self.assertRaises(ValueError):self.registry.create({'name':'螺母族'})

    def test_category_label_never_becomes_quality_or_reference_approval(self):
        result=self.registry.samples('screw');sample=result['samples'][0]
        self.assertEqual(sample['quality_label'],'unverified');self.assertFalse(sample['eligible_normal_reference'])
        self.assertNotIn('path',sample);self.assertEqual(self.registry.list()[0]['eligible_normal_references'],0)

    def test_quality_approval_rejects_whole_transaction(self):
        with self.assertRaisesRegex(ValueError,'不能确认正常'):
            self.registry.review('screw',{'sample_keys':[s['key'] for s in self.samples],'label':'normal','reviewer':'tester','use_for_bank':True})
        self.assertEqual(self.registry.reviews('screw'),{})
        self.assertEqual(self.registry.checks('screw'),{})

    def test_appearance_check_is_audited_without_quality_or_model_changes(self):
        key=self.samples[0]['key'];active=self.root/'perception/family_active.json';write(active,{'directory':'screw_family_v2'})
        before=active.read_bytes();manifest=(self.root/'product_families/screw/appearance.json').read_bytes()
        result=self.registry.review('screw',{'sample_keys':[key],'appearance_status':'accepted','reviewer':'tester'})
        self.assertFalse(result['model_changed']);self.assertEqual(before,active.read_bytes())
        self.assertFalse(result['quality_label_changed']);self.assertFalse(result['reference_approved'])
        sample=self.registry.samples('screw',status='accepted')['samples'][0]
        self.assertEqual(sample['quality_label'],'unverified');self.assertFalse(sample['eligible_normal_reference'])
        self.assertEqual(self.registry.samples('screw',status='normal')['count'],0)
        self.assertEqual(len(list((self.root/'product_families/screw/appearance_check_audit').glob('*.json'))),1)
        self.assertEqual(manifest,(self.root/'product_families/screw/appearance.json').read_bytes())

    def test_cross_family_model_and_outside_image_paths_rejected(self):
        family=self.registry.create({'name':'螺母族'});artifact=self.root/'experiments/candidate';artifact.mkdir(parents=True)
        write(artifact/'family.json',{'family_key':'screw','version':'screw-family-v2'})
        with self.assertRaisesRegex(ValueError,'所属'):self.registry.bind(family['key'],'experiments/candidate')
        with self.assertRaises(ValueError):self.registry.bind(family['key'],'../outside')
        data=self.registry.appearance('screw');data['samples'][0]['path']='perception/family_active.json';write(self.root/'product_families/screw/appearance.json',data)
        with self.assertRaisesRegex(ValueError,'路径'):self.registry.image('screw',self.samples[0]['key'])

    def test_defect_label_cannot_be_assigned_from_appearance_check(self):
        with self.assertRaisesRegex(ValueError,'不能确认正常'):
            self.registry.review('screw',{'sample_keys':[self.samples[0]['key']],'label':'defect','reviewer':'tester','use_for_bank':True})
        result=self.registry.review('screw',{'sample_keys':[self.samples[2]['key']],'appearance_status':'rejected','reviewer':'tester','note':'裁剪截断'})
        self.assertTrue(result['ok']);sample=self.registry.samples('screw',status='rejected')['samples'][0]
        self.assertEqual(sample['split'],'test');self.assertEqual(sample['quality_label'],'unverified')
        self.assertEqual(self.registry.samples('screw',status='defect')['count'],0)

    def test_legacy_quality_marks_are_retained_but_never_grant_normal_reference(self):
        path=self.root/'product_families/screw/appearance_reviews.json'
        write(path,{self.samples[0]['key']:{'label':'normal','use_for_bank':True,'reviewer':'legacy'}});before=path.read_bytes()
        sample=self.registry.samples('screw')['samples'][0];family=self.registry.list()[0]
        self.assertTrue(sample['legacy_quality_review_present']);self.assertFalse(sample['eligible_normal_reference'])
        self.assertEqual(sample['quality_label'],'unverified');self.assertEqual(family['reviewed_normal'],0)
        self.assertEqual(family['legacy_quality_marks'],1);self.assertEqual(family['eligible_normal_references'],0)
        self.registry.review('screw',{'sample_keys':[self.samples[0]['key']],'appearance_status':'accepted','reviewer':'tester'})
        self.assertEqual(before,path.read_bytes())

    def test_invalid_batch_cannot_save_a_partial_appearance_check(self):
        with self.assertRaisesRegex(ValueError,'不存在'):
            self.registry.review('screw',{'sample_keys':[self.samples[0]['key'],'absent'],'appearance_status':'accepted','reviewer':'tester'})
        self.assertEqual(self.registry.checks('screw'),{})
        with self.assertRaisesRegex(ValueError,'状态'):
            self.registry.review('screw',{'sample_keys':[self.samples[0]['key']],'appearance_status':'normal','reviewer':'tester'})

    def test_http_family_selection_image_review_and_error_contract(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from web import server
        with patch.object(server,'RESULTS_DIR',str(self.root)):
            client=TestClient(server.app)
            self.assertEqual(client.get('/api/product-families').json()['families'][0]['key'],'screw')
            draft=client.post('/api/product-families',json={'name':'网格族'}).json()['family']['key']
            self.assertFalse(client.get('/api/product-family/benchmark',params={'family':draft}).json()['ready'])
            result=client.get('/api/product-families/screw/appearance').json()
            self.assertEqual(client.get(result['samples'][0]['image_url']).content,b'image')
            payload={'sample_keys':[self.samples[0]['key']],'appearance_status':'accepted','reviewer':'API-test'}
            self.assertFalse(client.post('/api/product-families/screw/appearance/review',json=payload).json()['model_changed'])
            self.assertEqual(client.get('/api/product-families/screw/appearance?status=accepted').json()['count'],1)
            payload['label']='normal';payload['use_for_bank']=True
            self.assertEqual(client.post('/api/product-families/screw/appearance/review',json=payload).status_code,400)
            self.assertEqual(client.get('/api/product-families/absent/appearance').status_code,404)
            self.assertEqual(client.get('/api/product-families/screw/appearance',params={'limit':500}).status_code,400)


if __name__=='__main__':unittest.main()
