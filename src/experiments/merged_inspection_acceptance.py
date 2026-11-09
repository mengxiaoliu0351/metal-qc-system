"""Verify merged object metadata and a durable mixed-view screw batch over real HTTP."""
import argparse
from pathlib import Path
import time
import requests

from products.families import read,write

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    session=requests.Session();session.trust_env=False;base=args.base.rstrip('/')
    def get(path):
        r=session.get(base+path,timeout=60);r.raise_for_status();return r.json()
    def post(path,data):
        r=session.post(base+path,json=data,timeout=60);r.raise_for_status();return r.json()
    data=get('/api/products');targets={x['key']:x for x in data['inspection_targets']}
    assert {'screw','metal_nut','grid'}<=targets.keys() and 'product_a269405b7b9a' not in targets
    assert targets['screw']['mode']=='family' and targets['screw']['batch_key']=='family:screw' and len(data['products'])>=4
    folder=ROOT/'results/experiments/screw_family_router_v1';samples=read(folder/'demo_samples.json')['samples'];samples=[s for s in samples if s['expected_view'] is not None]
    job=post('/api/batches',{'product':'family:screw','purpose':'simulation','name':'合并检测对象接口验收'})
    assert job['recipe']['config']['mode']=='family' and set(job['recipe']['destinations'])=={'screw','product_a269405b7b9a'}
    for sample in samples:
        with (folder/sample['file']).open('rb') as image:
            r=session.post(base+f"/api/batches/{job['id']}/items",files={'file':(sample['id'][:12]+'.png',image,'image/png')},timeout=60);r.raise_for_status()
    post(f"/api/batches/{job['id']}/start",{});deadline=time.monotonic()+240
    while True:
        detail=get('/api/batches/'+job['id'])
        if detail['status'] not in ('queued','running'):break
        if time.monotonic()>deadline:raise TimeoutError('Mixed-view batch did not finish within acceptance bound')
        time.sleep(1)
    assert detail['status']=='completed',detail['status'];assert detail['counts']['completed']==3
    results=[]
    for item in detail['items']:
        result=item['result'];assert result['family_key']=='screw' and result['family_routing']['matched']
        if result['family_routing']['view_key']=='m2_body':assert result['quality_model_bound'] is False and result['score'] is None
        else:assert result['family_action']=='product_workflow'
        r=session.get(base+'/api/report/'+result['session_id'],timeout=60);r.raise_for_status();assert r.content.startswith(b'%PDF')
        results.append({'view':result['family_routing']['view_key'],'action':result['family_action'],'quality_model_bound':result['quality_model_bound'],
            'session_id':result['session_id'],'pdf_valid':True,'requires_review':result['disposition']['requires_review']})
    assert {r['view'] for r in results}=={'mvtec_screw','m2_head','m2_body'}
    write(args.output,{'passed':True,'base':base,'batch_id':job['id'],'single_object_choices':list(targets),'mixed_view_results':results,
        'internal_head_recipe_hidden':True,'scope':'one simulation batch with three capture views; no human quality labels submitted'})
    print({'passed':True,'mixed_views':3,'family_batch_completed':True,'internal_head_recipe_hidden':True},flush=True)

if __name__=='__main__':main()
