"""Compare the real appearance HTTP route with frozen offline predictions."""
import argparse
import json
from pathlib import Path
import time

import requests
import numpy as np

from perception.appearance import AppearanceMatcher
from perception.appearance_boundary import load_matcher
from products.families import read,write

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--experiment',type=Path,default=ROOT/'results/experiments/screw_appearance_v1');parser.add_argument('--all',action='store_true');args=parser.parse_args()
    folder=args.experiment;p=read(folder/'protocol.json');model=load_matcher(folder/'appearance.json')
    partition='fresh_test' if any(s['role']=='fresh_test' for s in p['samples']) else 'test'
    selected=[]
    for category in (5,6,12,4):
        chosen={}
        for sample in p['samples']:
            if sample['role']!=partition or sample['category_id']!=category:continue
            with np.load(folder/'features'/(sample['key']+'.npz'),allow_pickle=False) as data:pred=model.predict(data[model.meta['selected_method']])
            state='matched' if pred['matched'] else 'uncertain'
            if state not in chosen:chosen[state]=sample
            if len(chosen)==2:break
        selected.extend(chosen.values())
    selected.extend([s for s in p['samples'] if s['role']=='control'][::10])
    if args.all:selected=[s for s in p['samples'] if s['role']==partition]
    session=requests.Session();session.trust_env=False;checks=[]
    benchmark=session.get(args.base+'/api/product-families/screw/matching',timeout=40);benchmark.raise_for_status();d=benchmark.json()
    assert d['ready'] and d['model']['arrays_sha256']==model.meta['arrays_sha256']
    for sample in selected:
        with np.load(folder/'features'/(sample['key']+'.npz'),allow_pickle=False) as data:expected=model.predict(data[model.meta['selected_method']])
        began=time.monotonic()
        with (ROOT/sample['path']).open('rb') as f:r=session.post(args.base+'/api/product-families/screw/matching/detect',files={'file':('part.png',f,'image/png')},timeout=120)
        if args.all and r.status_code==400:
            assert not expected['matched'],'Input gate unexpectedly rejected an accepted offline sample'
            checks.append({'key':sample['key'],'true_category':sample['category_id'],'matched_category':None,'status':400,'reason':'input_quality_gate','similarity_error':0.,'seconds':round(time.monotonic()-began,3)})
            continue
        r.raise_for_status();actual=r.json()
        assert actual['category_id']==expected['category_id'] and actual['matched']==expected['matched'],sample['key']
        assert abs(actual['similarity']-expected['similarity'])<1e-4
        assert actual['quality_assessment']=='not_available' and actual['automatic_archival'] is False
        assert not {'normal','defect','quality_label'}&set(actual)
        for ref in actual['references'] if not args.all or len(checks)<3 else []:
            image=session.get(args.base+ref['image_url'],timeout=30);image.raise_for_status();assert image.headers['content-type'].startswith('image/')
        if model.meta.get('matcher_kind') and (not args.all or len(checks)<3):
            assert actual['exclusion_reference'] is not None
            response=session.get(args.base+actual['exclusion_reference']['image_url'],timeout=30);response.raise_for_status()
        checks.append({'key':sample['key'],'true_category':sample['category_id'],'matched_category':actual['category_id'],'similarity_error':abs(actual['similarity']-expected['similarity']),'seconds':round(time.monotonic()-began,3)})
        if args.all and len(checks)%40==0:print({'website_images_checked':len(checks),'total':len(selected)},flush=True)
    write(args.output,{'base':args.base,'version':model.meta['version'],'partition':partition,'checks':checks,'passed':True,'quality_labels_created':False,
        'full_partition':args.all,'scope':'full local HTTP partition replay' if args.all else 'selected acceptance images, not a full website replay of the retained crops'})
    print(json.dumps({'passed':True,'images':len(checks),'input_quality_rejects':sum(c.get('status')==400 for c in checks),'max_similarity_error':max(c['similarity_error'] for c in checks)}))


if __name__=='__main__':main()
