"""Frozen human-reviewed reference retrieval with grouped calibration/test frames."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from perception.appearance import full_object,color_shape,ranked,AppearanceMatcher
from products.families import read,write

ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output):
    manifest_path=ROOT/'results/product_families/screw/appearance.json';manifest=read(manifest_path)
    snapshot=read(output/'review_snapshot.json');accepted={r['key'] for r in snapshot['reviews'] if r['appearance_status']=='accepted'}
    references=[s for s in manifest['samples'] if s['key'] in accepted]
    if len(references)!=len(accepted) or any(s['split']!='train' for s in references):raise ValueError('Only reviewed train references may be fitted')
    reserved_path=ROOT/'results/product_families/screw/appearance_reservations.json'
    reserved=read(reserved_path).get('samples',{}) if reserved_path.exists() else {}
    reserved_frames={s['source_frame'] for s in reserved.values()}
    if any(s['key'] in reserved or s.get('source_frame') in reserved_frames for s in references):raise ValueError('Frozen evaluation references cannot be fitted')
    categories=sorted({s['category_id'] for s in references})
    if len(categories)<2 or any(sum(s['category_id']==c for s in references)<2 for c in categories):raise ValueError('Insufficient category references')
    samples=[{**s,'path':'results/'+s['path'],'role':'reference' if s['key'] in accepted else s['split']}
        for s in manifest['samples'] if s['key'] in accepted or s['split'] in ('validation','test')]
    for field in ('source_frame','pixel_hash'):
        roles={}
        for s in samples:roles.setdefault(s[field],set()).add(s['role'])
        if any(len(values)>1 for values in roles.values()):raise ValueError('Reference/calibration/test grouping leak: '+field)
    controls=[]
    for product in ('metal_nut','grid'):
        for path in sorted((ROOT/f'data/mvtec/{product}/test/good').glob('*.png'))[:10]:
            controls.append({'key':sha(path),'path':path.relative_to(ROOT).as_posix(),'file_sha256':sha(path),
                'category_id':-1,'role':'control','source_frame':f'{product}:{path.name}','source_product':product})
    protocol={'version':'screw-appearance-v1','seed':20261106,'methods':['color_shape','clip'],
        'reference_categories':[t for t in manifest['types'] if t['category_id'] in categories],
        'samples':samples+controls,'review_snapshot_sha256':sha(output/'review_snapshot.json'),'manifest_sha256':sha(manifest_path),
        'weights_sha256':sha(ROOT/'weights/CLIP-ViT-B-16-open_clip_model.safetensors'),
        'extractor_sha256':sha(ROOT/'src/perception/appearance.py'),
        'reference_counts':dict(Counter(str(s['category_id']) for s in references)),
        'fit':'nearest human-reviewed reference within each category; one combined reference file',
        'selection':'method and global similarity/margin gate selected on validation only; maximize .5*known_correct_acceptance + .5*other_category_rejection; ties prefer known coverage',
        'gate_grid':{'score_quantiles':21,'margin':[0,.005,.01,.02,.04,.08,.12]},
        'scope':'provided annotation crops from one controlled dataset; category labels only; no physical identities or new quality labels; controls are earlier public product images',
        'quality_labels_used':False,'automatic_archival':False}
    path=output/'protocol.json'
    if path.exists() and read(path)!=protocol:raise ValueError('Protocol is frozen; use a new version')
    write(path,protocol);print(json.dumps({'roles':dict(Counter(s['role'] for s in protocol['samples'])),'references':protocol['reference_counts']}),flush=True)


def extract(output):
    protocol=read(output/'protocol.json');cache=output/'features';cache.mkdir(exist_ok=True)
    if sha(ROOT/'weights/CLIP-ViT-B-16-open_clip_model.safetensors')!=protocol['weights_sha256'] or sha(ROOT/'src/perception/appearance.py')!=protocol['extractor_sha256']:raise ValueError('Frozen extractor changed')
    missing=[s for s in protocol['samples'] if not (cache/(s['key']+'.npz')).exists()]
    if not missing:return
    import torch
    from perception.winclip_patch import WinCLIPPatch
    torch.set_num_threads(2);model=WinCLIPPatch(device='cpu');began=time.monotonic()
    for start in range(0,len(missing),4):
        batch=missing[start:start+4];images=[]
        for s in batch:
            if sha(ROOT/s['path'])!=s['file_sha256']:raise ValueError('Frozen candidate changed')
            with Image.open(ROOT/s['path']) as im:images.append(im.convert('RGB'))
        _,vectors=model.extract_batch([full_object(im) for im in images])
        for s,im,vector in zip(batch,images,vectors):
            path=cache/(s['key']+'.npz');temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as f:np.savez_compressed(f,clip=vector,color_shape=color_shape(im))
            temporary.replace(path)
        if start%40==0:print(json.dumps({'features':start+len(batch),'total':len(missing),'seconds':round(time.monotonic()-began)}),flush=True)


def stats(rows,pred,categories,gate):
    truth=np.array([s['category_id'] for s in rows]);known=np.isin(truth,categories)
    accept=(pred['similarity']>=gate['similarity'])&(pred['margin']>=gate['margin']);correct=pred['category']==truth
    ratio=lambda n,d:float(n/d) if d else None
    return {'n':len(rows),'known':int(known.sum()),'other':int((~known).sum()),
        'known_top1_accuracy':ratio((known&correct).sum(),known.sum()),'known_coverage':ratio((known&accept).sum(),known.sum()),
        'known_correct_acceptance':ratio((known&accept&correct).sum(),known.sum()),
        'accepted_precision_all':ratio((known&accept&correct).sum(),accept.sum()),
        'other_rejection':ratio((~known&~accept).sum(),(~known).sum()),
        'accepted_known_wrong':int((known&accept&~correct).sum()),'accepted_other':int((~known&accept).sum()),
        'by_category':{str(c):{'n':int((truth==c).sum()),'correct_top1':int(((truth==c)&correct).sum()),
            'accepted_correct':int(((truth==c)&accept&correct).sum()),'accepted_wrong':int(((truth==c)&accept&~correct).sum()),
            'rejected':int(((truth==c)&~accept).sum())} for c in sorted(set(truth.tolist()))}}


def fit_evaluate(output):
    if (output/'report.json').exists():raise ValueError('Completed report is frozen; do not refit against retained test')
    p=read(output/'protocol.json');refs=[s for s in p['samples'] if s['role']=='reference'];cal=[s for s in p['samples'] if s['role']=='validation']
    categories=[c['category_id'] for c in p['reference_categories']];params={'labels':np.array([s['category_id'] for s in refs])};results={};choices=[]
    def features(rows,method):
        values=[]
        for s in rows:
            with np.load(output/'features'/(s['key']+'.npz'),allow_pickle=False) as data:values.append(data[method])
        return np.asarray(values)
    for method in p['methods']:
        params[method]=features(refs,method);pred=ranked(features(cal,method),params[method],params['labels'],categories)
        best=None
        for score in np.unique(np.r_[-1.,np.quantile(pred['similarity'],np.linspace(0,1,p['gate_grid']['score_quantiles'])),1.00001]):
            for margin in p['gate_grid']['margin']:
                gate={'similarity':float(score),'margin':float(margin)};m=stats(cal,pred,categories,gate)
                objective=.5*m['known_correct_acceptance']+.5*m['other_rejection']
                key=(objective,m['known_coverage'],-margin,-float(score))
                if best is None or key>best[0]:best=(key,gate,m)
        results[method]={'gate':best[1],'validation':best[2],'validation_objective':best[0][0]};choices.append((best[0][0],best[2]['known_coverage'],method))
    method=max(choices)[2];np.savez_compressed(output/'appearance.npz',**params)
    metadata={'version':p['version'],'family_key':'screw','arrays':'appearance.npz','arrays_sha256':sha(output/'appearance.npz'),
        'protocol_sha256':sha(output/'protocol.json'),'weights_sha256':p['weights_sha256'],'categories':p['reference_categories'],
        'reference_keys':[s['key'] for s in refs],'reference_counts':p['reference_counts'],'selected_method':method,'gate':results[method]['gate'],
        'quality_labels_used':False,'automatic_archival':False,'input':'single cropped part; full-object square padding, no automatic localization'}
    write(output/'appearance.json',metadata)
    # Test is first read for scoring only AFTER all method/gate choices are fixed.
    for arm in p['methods']:
        for role in ('test','control'):
            rows=[s for s in p['samples'] if s['role']==role]
            pred=ranked(features(rows,arm),params[arm],params['labels'],categories)
            results[arm][role]=stats(rows,pred,categories,results[arm]['gate'])
    model=AppearanceMatcher(output/'appearance.json');rows=[s for s in p['samples'] if s['role']=='test'];parity=features(rows[:8],method)
    for vector in parity:assert model.predict(vector)['quality_assessment']=='not_available'
    report={'version':p['version'],'selected_method':method,'reference_counts':p['reference_counts'],'methods':results,
        'arrays_sha256':metadata['arrays_sha256'],'protocol_sha256':metadata['protocol_sha256'],'quality_labels_used':False,
        'automatic_archival':False,'scope':p['scope'],'reviewed_reference_count':len(refs),'physical_ids_available':False}
    write(output/'report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','extract','evaluate']);parser.add_argument('--output',type=Path,default=ROOT/'results/experiments/screw_appearance_v1');args=parser.parse_args()
    {'prepare':prepare,'extract':extract,'evaluate':fit_evaluate}[args.action](args.output)


if __name__=='__main__':main()
