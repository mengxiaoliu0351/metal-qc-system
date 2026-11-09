"""Freeze and validate three-view suitability routing, separately from quality models."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.m2_body_quality import ROOT,save_arrays
from experiments.prepare_m2_body import sha
from perception.appearance import full_object,unit
from perception.family_router import FamilyRouter
from products.families import read,write
from review.workflow import image_hash

OUT=ROOT/'results/experiments/screw_family_router_v1'
VIEWS=[{'key':'mvtec_screw','name':'螺丝近景视图','product_key':'screw'},
    {'key':'m2_head','name':'M2头部俯视','product_key':'product_a269405b7b9a'},
    {'key':'m2_body','name':'M2螺杆侧视','product_key':None}]


def prepare(out):
    family=read(ROOT/'results/experiments/screw_family_v2/protocol.json');body=read(ROOT/'data/external/m2_v2/body-manifest.json');appearance=read(ROOT/'results/product_families/screw/appearance.json');rows=[]
    for g in (0,1,2):
        source=family['samples'] if g<2 else body['images']
        for role,original,n in [('reference','bank',80),('calibration','calibration',40),('test','test',40)]:
            candidates=[s for s in source if (g==2 or s['group']==g) and s['role']==original and not s.get('excluded')]
            candidates=sorted(candidates,key=lambda s:s['pixel_hash'])[:n]
            if len(candidates)!=n:raise ValueError('Insufficient frozen view images')
            for s in candidates:
                path=s.get('path',s.get('source_path')).replace('\\','/')
                rows.append({'path':path,'pixel_hash':s['pixel_hash'],'file_sha256':sha(ROOT/path),'view':g,'role':role,'kind':'known_view'})
    for split,role in [('validation','calibration'),('test','test')]:
        for category in sorted({s['category_id'] for s in appearance['samples']}):
            selected=sorted((s for s in appearance['samples'] if s['split']==split and s['category_id']==category),key=lambda s:s['pixel_hash'])[:5]
            for s in selected:rows.append({'path':'results/'+s['path'],'pixel_hash':s['pixel_hash'],'file_sha256':s['file_sha256'],'view':-1,'role':role,'kind':'unverified_appearance'})
    for product in ('metal_nut','grid'):
        paths=sorted((ROOT/f'data/mvtec/{product}/train/good').glob('*.png'))[:20]
        if len(paths)!=20:raise ValueError('Missing non-screw control images')
        for i,path in enumerate(paths):rows.append({'path':path.relative_to(ROOT).as_posix(),'pixel_hash':image_hash(path),'file_sha256':sha(path),'view':-1,
            'role':'exclusion' if i<5 else 'calibration' if i<10 else 'test','kind':'other_product'})
    identities={}
    for s in rows:identities.setdefault(s['pixel_hash'],set()).add(s['role'])
    if any(len(v)>1 for v in identities.values()):raise ValueError('Routing roles overlap')
    meta=read(ROOT/'results/experiments/screw_appearance_v3/appearance.json')
    p={'version':'screw-family-router-v1','family_key':'screw','views':VIEWS,'samples':rows,
        'weights_sha256':sha(ROOT/'weights/CLIP-ViT-B-16-open_clip_model.safetensors'),
        'preprocessing_sha256':sha(ROOT/'src/perception/appearance.py'),'extractor_sha256':sha(ROOT/'src/perception/winclip_patch.py'),
        'exclusion_arrays_sha256':meta['arrays_sha256'],'exclusion_reference':'existing 24 reviewed appearance references + 72 source-category exclusions, none quality-labelled',
        'gate_rule':'per-view known calibration 1st percentile similarity minus .01 and margin minus .005 (floor .01); nearest exclusion must not win',
        'release_gate':{'minimum_known_coverage':.85,'minimum_accepted_routing_accuracy':.99,'maximum_unknown_acceptance':.05},
        'calibration_gate':{'minimum_known_coverage':.9,'minimum_accepted_routing_accuracy':.99,'maximum_unknown_acceptance':.05},
        'evaluation_scope':'existing public quality/appearance partitions reused for routing integration regression; not a new factory or physical-item blind test',
        'quality_decision_permission':False}
    if (out/'protocol.json').exists() and read(out/'protocol.json')!=p:raise ValueError('Routing protocol is frozen')
    write(out/'protocol.json',p);print({'routing_roles':dict(Counter(s['role'] for s in rows))},flush=True)
    reservation_path=ROOT/'results/evaluation_reservations.json';reserved=read(reservation_path)
    for s in rows:
        if s['role']=='test':reserved.setdefault('global',{}).setdefault(s['pixel_hash'],{'experiment':p['version'],'role':'routing_integration_test','scope':'routing regression only, not quality accuracy'})
    write(reservation_path,reserved)


def extract(out):
    p=read(out/'protocol.json');cache=out/'features';cache.mkdir(exist_ok=True)
    for path,key in [(ROOT/'weights/CLIP-ViT-B-16-open_clip_model.safetensors','weights_sha256'),(ROOT/'src/perception/appearance.py','preprocessing_sha256'),(ROOT/'src/perception/winclip_patch.py','extractor_sha256')]:
        if sha(path)!=p[key]:raise ValueError('Routing implementation changed')
    from perception.winclip_patch import WinCLIPPatch
    import torch
    torch.set_num_threads(2);detector=WinCLIPPatch(device='cpu');missing=[s for s in p['samples'] if not (cache/(s['pixel_hash']+'.npz')).exists()]
    for start in range(0,len(missing),8):
        batch=missing[start:start+8];images=[]
        for s in batch:
            if sha(ROOT/s['path'])!=s['file_sha256']:raise ValueError('Routing source changed')
            with Image.open(ROOT/s['path']) as im:images.append(full_object(im))
        _,vectors=detector.extract_batch(images)
        for s,vector in zip(batch,vectors):save_arrays(cache/(s['pixel_hash']+'.npz'),vector=vector)
        if start%80==0:print({'routing_features':start+len(batch),'total':len(missing)},flush=True)


def features(out,rows):
    return np.array([np.load(out/'features'/(s['pixel_hash']+'.npz'),allow_pickle=False)['vector'] for s in rows])


def metrics(model,out,rows):
    predictions=[model.predict(x) for x in features(out,rows)];known=[(s,r) for s,r in zip(rows,predictions) if s['view']>=0];unknown=[r for s,r in zip(rows,predictions) if s['view']<0]
    accepted=[(s,r) for s,r in known if r['matched']];correct=sum(r['view_key']==VIEWS[s['view']]['key'] for s,r in accepted)
    return {'known':len(known),'known_accepted':len(accepted),'known_coverage':len(accepted)/len(known),
        'accepted_routing_accuracy':correct/len(accepted) if accepted else 0,'known_wrong_accepted':len(accepted)-correct,
        'unknown':len(unknown),'unknown_accepted':sum(r['matched'] for r in unknown),'unknown_acceptance':sum(r['matched'] for r in unknown)/len(unknown),
        'by_view':{view['key']:{'n':sum(s['view']==i for s,_ in known),'accepted_correct':sum(s['view']==i and r['matched'] and r['view_key']==view['key'] for s,r in known),
            'coverage':sum(s['view']==i and r['matched'] for s,r in known)/sum(s['view']==i for s,_ in known)} for i,view in enumerate(VIEWS)}}


def passes(m,g):return m['known_coverage']>=g['minimum_known_coverage'] and all(row['coverage']>=g['minimum_known_coverage'] for row in m['by_view'].values()) and m['accepted_routing_accuracy']>=g['minimum_accepted_routing_accuracy'] and m['unknown_acceptance']<=g['maximum_unknown_acceptance']


def fit(out):
    if (out/'router.json').exists():raise ValueError('Do not refit routing after validation')
    p=read(out/'protocol.json');centroids=[]
    for i in range(3):centroids.append(unit(features(out,[s for s in p['samples'] if s['role']=='reference' and s['view']==i]).mean(axis=0)))
    centroids=np.array(centroids);a=ROOT/'results/experiments/screw_appearance_v3/appearance.npz'
    if sha(a)!=p['exclusion_arrays_sha256']:raise ValueError('Exclusion version changed')
    with np.load(a,allow_pickle=False) as z:exclusions=np.r_[z['clip'],z['exclusion_clip']]
    exclusions=np.r_[exclusions,features(out,[s for s in p['samples'] if s['role']=='exclusion'])];gates=[]
    for i in range(3):
        x=unit(features(out,[s for s in p['samples'] if s['role']=='calibration' and s['view']==i]));scores=x@centroids.T
        others=np.max(np.delete(scores,i,axis=1),axis=1);gates.append({'similarity':float(max(.75,np.quantile(scores[:,i],.01)-.01)),
            'margin':float(max(.01,np.quantile(scores[:,i]-others,.01)-.005))})
    save_arrays(out/'router.npz',centroids=centroids,exclusions=exclusions)
    meta={k:p[k] for k in ('version','family_key','views','weights_sha256','preprocessing_sha256','extractor_sha256','quality_decision_permission')}
    meta.update(arrays='router.npz',arrays_sha256=sha(out/'router.npz'),protocol_sha256=sha(out/'protocol.json'),gates=gates)
    write(out/'router.json',meta);model=FamilyRouter(out/'router.json');m=metrics(model,out,[s for s in p['samples'] if s['role']=='calibration'])
    write(out/'calibration_report.json',{'metrics':m,'passed':passes(m,p['calibration_gate']),'test_used_for_fitting':False});print({'routing_calibration':m},flush=True)


def evaluate(out):
    if (out/'report.json').exists():raise ValueError('Routing validation is frozen')
    p=read(out/'protocol.json');model=FamilyRouter(out/'router.json');rows=[s for s in p['samples'] if s['role']=='test'];m=metrics(model,out,rows);cal=read(out/'calibration_report.json')
    ready=cal['passed'] and passes(m,p['release_gate']);report={'version':p['version'],'metrics':m,'calibration':cal,'operational_routing_ready':ready,
        'arrays_sha256':model.meta['arrays_sha256'],'protocol_sha256':model.meta['protocol_sha256'],'quality_decision_permission':False,'scope':p['evaluation_scope']}
    write(out/'report.json',report)
    examples=[];selected=[next(s for s in rows if s['view']==i) for i in range(3)]
    selected+=[next(s for s in rows if s['kind']=='unverified_appearance'),next(s for s in rows if s['kind']=='other_product')]
    for s in selected:
        target=out/'demo_images'/(s['pixel_hash']+'.png');target.parent.mkdir(exist_ok=True)
        with Image.open(ROOT/s['path']) as image:image.convert('RGB').save(target)
        examples.append({'id':s['pixel_hash'],'file':target.relative_to(out).as_posix(),'file_sha256':sha(target),
            'label':VIEWS[s['view']]['name'] if s['view']>=0 else '未验证外观' if s['kind']=='unverified_appearance' else '其他产品',
            'expected_view':VIEWS[s['view']]['key'] if s['view']>=0 else None,'selection':'first frozen hash per view/control, not selected for correct output'})
    write(out/'demo_samples.json',{'samples':examples,'scope':p['evaluation_scope']});print({'routing_validation':m,'routing_ready':ready},flush=True)


def references(out):
    p=read(out/'protocol.json');items=[]
    for i,view in enumerate(VIEWS):
        s=next(row for row in p['samples'] if row['role']=='reference' and row['view']==i);target=out/'reference_images'/(view['key']+'.png');target.parent.mkdir(exist_ok=True)
        with Image.open(ROOT/s['path']) as image:
            image=image.convert('RGB');image.thumbnail((512,512),Image.Resampling.LANCZOS);image.save(target)
        items.append({'view_key':view['key'],'file':target.relative_to(out).as_posix(),'file_sha256':sha(target),'source_pixel_hash':s['pixel_hash'],
            'source':'https://www.mvtec.com/research-teaching/datasets/mvtec-ad' if i==0 else 'https://data.mendeley.com/datasets/2y6mcsyrpy/2',
            'purpose':'normal-bank reference illustrating capture view; resized preview, not quality truth for uploads'})
    write(out/'references.json',{'items':items})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','extract','fit','evaluate','references']);parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    {'prepare':prepare,'extract':extract,'fit':fit,'evaluate':evaluate,'references':references}[args.action](args.output)


if __name__=='__main__':main()
