"""Prespecified normal-only, labelled and out-of-fold fusion BODY comparison."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
from PIL import Image

from experiments.prepare_m2_body import sha
from perception.appearance import full_object,unit
from perception.body_preprocess import prepare_body
from perception.body_quality import body_scores,scope_values,BodyQualityModel
from perception.feature_backends import representative_bank,cosine_anomalies,binary_probability,export_binary
from products.families import read,write
from review.workflow import calibrate,measurement

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'results/experiments/m2_body_quality_v1'
ARMS=('normal_memory','global_binary','fast_fusion')


def save_arrays(path,**values):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,**values)
    temporary.replace(path)


def prepare(output):
    manifest=ROOT/'data/external/m2_v2/body-manifest.json';body=read(manifest);head=read(ROOT/'data/external/m2_v2/head-manifest.json')
    samples=[{**s,'path':s['source_path'],'view':'body'} for s in body['images'] if not s['excluded'] and s['role']!='feedback']
    for s in head['images']:
        if s['excluded'] or s['role'] not in ('bank','calibration'):continue
        samples.append({**s,'path':s['source_path'].replace('\\','/'),'view':'head','role':'head_reference' if s['role']=='bank' else 'head_calibration','file_sha256':sha(ROOT/s['source_path'])})
    for label in ('normal','defect'):
        control=sorted((s for s in head['images'] if not s['excluded'] and s['role']=='test' and s['label']==label),key=lambda s:s['pixel_hash'])[:50]
        for s in control:samples.append({**s,'path':s['source_path'].replace('\\','/'),'view':'head','role':'head_control','file_sha256':sha(ROOT/s['source_path'])})
    partitions={}
    for s in samples:partitions.setdefault(s['pixel_hash'],set()).add(s['role'])
    if any(len(parts)>1 for parts in partitions.values()):raise ValueError('Cross-role exact image leak')
    p={'version':'m2-body-quality-v1','dataset':body['dataset'],'body_manifest_sha256':sha(manifest),'head_manifest_sha256':sha(ROOT/'data/external/m2_v2/head-manifest.json'),
        'samples':samples,'seed':20261106,'label_basis':'publisher OK/NG folders','physical_identity':'unknown; image-level only',
        'preprocessing':'unlabelled central component proposal with full-image fallback, then pad to 224; same frozen rule for every arm','bank_size':512,'classifier_C':.1,'fusion_oof_folds':5,
        'methods':list(ARMS),'selection':'highest balanced accuracy on BODY calibration only; ties prefer normal_memory then global_binary then fast_fusion',
        'normal_only_threshold':'99th percentile of normal calibration; no defect calibration labels',
        'labelled_thresholds':'existing calibration-only balanced-accuracy objective and high-risk precision hint; no test tuning',
        'view_gate':'BODY normal+defect calibration coverage >=95% each; maximize HEAD calibration rejection; no HEAD quality classification',
        'scope_margin_grid':[-.04,-.02,0,.02,.04,.08,.12],
        'qualification_gate':{'minimum_recall':.9,'maximum_false_positive_rate':.1,'minimum_scope_coverage':.95},
        'sealed_trial_gate':{'minimum_recall':.9,'maximum_false_positive_rate':.1,'minimum_scope_coverage':.95,'minimum_head_rejection':.95},
        'qualification_used_for_fitting':False,'test_used_for_fitting':False,'automatic_archival':False,
        'weights_sha256':sha(ROOT/'weights/CLIP-ViT-B-16-open_clip_model.safetensors'),
        'extractor_sha256':sha(ROOT/'src/perception/winclip_patch.py'),'preprocessing_sha256':sha(ROOT/'src/perception/body_preprocess.py')}
    target=output/'protocol.json'
    if target.exists() and read(target)!=p:raise ValueError('Protocol differs; do not overwrite frozen BODY experiment')
    write(target,p);write(output/'test_reservations.json',{'pixel_hashes':[s['pixel_hash'] for s in samples if s['role'] in ('test','layout_preview')],
        'primary_sealed_test_n':sum(s['role']=='test' for s in samples),'layout_preview_n':sum(s['role']=='layout_preview' for s in samples),
        'scope':'all official BODY test images, including separately reported layout previews; not feedback training'})
    reserved_path=ROOT/'results/evaluation_reservations.json'
    reserved=read(reserved_path) if reserved_path.exists() else {'version':1,'products':{}}
    for s in samples:
        if s['role'] in ('test','layout_preview'):reserved.setdefault('global',{}).setdefault(s['pixel_hash'],{'experiment':p['version'],'role':s['role'],'dataset':'M2 BODY V2'})
    write(reserved_path,reserved)
    print({'frozen_roles':dict(Counter(s['role'] for s in samples))},flush=True)


def extract(output):
    p=read(output/'protocol.json')
    if sha(ROOT/'weights/CLIP-ViT-B-16-open_clip_model.safetensors')!=p['weights_sha256'] or sha(ROOT/'src/perception/winclip_patch.py')!=p['extractor_sha256'] or sha(ROOT/'src/perception/body_preprocess.py')!=p['preprocessing_sha256']:raise ValueError('Frozen encoder changed')
    from perception.winclip_patch import WinCLIPPatch
    import torch
    torch.set_num_threads(2);detector=WinCLIPPatch(device='cpu');cache=output/'features';cache.mkdir(exist_ok=True);started=time.monotonic()
    bank_path=output/'normal_bank.npz';references=[s for s in p['samples'] if s['role']=='bank']
    if not bank_path.exists():
        parts=[];vectors=[]
        for start in range(0,len(references),4):
            batch=references[start:start+4];images=[]
            for s in batch:
                if sha(ROOT/s['path'])!=s['file_sha256']:raise ValueError('Frozen source changed')
                with Image.open(ROOT/s['path']) as im:images.append(prepare_body(im)[0])
            local,global_x=detector.extract_batch(images);parts.extend(local);vectors.extend(global_x)
        bank=representative_bank(np.concatenate(parts),size=p['bank_size'],seed=p['seed'])
        save_arrays(bank_path,bank=bank,body_centroid=unit(np.mean(vectors,axis=0)))
        for s,local,vector in zip(references,parts,vectors):save_arrays(cache/(s['pixel_hash']+'.npz'),vector=vector,raw=np.array(float(cosine_anomalies(local,bank).max())))
    with np.load(bank_path,allow_pickle=False) as z:bank=z['bank'].copy()
    development=[s for s in p['samples'] if s['role'] not in ('test','head_control','layout_preview')]
    missing=[s for s in development if not (cache/(s['pixel_hash']+'.npz')).exists()]
    for start in range(0,len(missing),4):
        batch=missing[start:start+4];images=[];regions=[]
        for s in batch:
            if sha(ROOT/s['path'])!=s['file_sha256']:raise ValueError('Frozen source changed')
            with Image.open(ROOT/s['path']) as im:
                prepared,region=prepare_body(im);images.append(prepared);regions.append(region)
        patches,vectors=detector.extract_batch(images)
        for s,local,vector,region in zip(batch,patches,vectors,regions):
            raw=float(cosine_anomalies(local,bank).max());save_arrays(cache/(s['pixel_hash']+'.npz'),vector=vector,raw=np.array(raw),roi_found=np.array(region['proposal_found']))
        if start%40==0:print({'development_features':start+len(batch),'total':len(missing),'seconds':round(time.monotonic()-started)},flush=True)


def load_features(output,rows):
    vectors=[];raw=[]
    for s in rows:
        with np.load(output/'features'/(s['pixel_hash']+'.npz'),allow_pickle=False) as z:vectors.append(z['vector']);raw.append(float(z['raw']))
    return np.asarray(vectors),np.asarray(raw)


def measure(labels,scores,low,high):
    from sklearn.metrics import roc_auc_score
    m=measurement(labels,scores,low,high);m['auroc']=float(roc_auc_score(labels,scores)) if len(set(labels))==2 else None
    return m


def fit(output):
    if (output/'quality.json').exists():raise ValueError('BODY model already frozen; do not refit after evaluation')
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold,cross_val_predict
    from threadpoolctl import threadpool_limits
    p=read(output/'protocol.json');train=[s for s in p['samples'] if s['role']=='train'];cal=[s for s in p['samples'] if s['role']=='calibration'];qual=[s for s in p['samples'] if s['role']=='qualification']
    x,raw=load_features(output,train);y=np.array([s['label']=='defect' for s in train],dtype=int)
    xc,rc=load_features(output,cal);yc=np.array([s['label']=='defect' for s in cal],dtype=int)
    xq,rq=load_features(output,qual);yq=np.array([s['label']=='defect' for s in qual],dtype=int)
    def classifier():return Pipeline([('sc',StandardScaler()),('clf',LogisticRegression(C=p['classifier_C'],class_weight='balanced',max_iter=2000,random_state=p['seed']))])
    with threadpool_limits(limits=2):
        global_model=classifier();oof=cross_val_predict(global_model,x.astype(float),y,cv=StratifiedKFold(p['fusion_oof_folds'],shuffle=True,random_state=p['seed']),method='predict_proba')[:,1]
        global_model.fit(x.astype(float),y);fusion=classifier();fusion.fit(np.c_[raw,oof],y)
    params=export_binary(global_model,'global');params.update(export_binary(fusion,'fusion'))
    with np.load(output/'normal_bank.npz',allow_pickle=False) as z:params.update({k:z[k].copy() for k in z.files})
    headrefs=[s for s in p['samples'] if s['role']=='head_reference'];xh,_=load_features(output,headrefs);params['head_centroid']=unit(xh.mean(axis=0))
    headcal=[s for s in p['samples'] if s['role']=='head_calibration'];xhc,_=load_features(output,headcal)
    body_cos,_,body_gap=scope_values(xc,params);head_cos,_,head_gap=scope_values(xhc,params)
    minimum=float(max(-1,np.quantile(body_cos,.01)-.03));scope_options=[]
    for margin in p['scope_margin_grid']:
        keep=(body_cos>=minimum)&(body_gap>=margin)
        if any(np.mean(keep[yc==label])<.95 for label in (0,1)):continue
        reject=~((head_cos>=minimum)&(head_gap>=margin));scope_options.append((float(reject.mean()),float(keep.mean()),-abs(margin),margin))
    scope_fitted=bool(scope_options);margin=float(max(scope_options)[3]) if scope_options else 2.
    thresholds={};calibration={};qualification={};choices=[]
    for i,arm in enumerate(ARMS):
        scores=body_scores(xc,rc,params,arm)
        if arm=='normal_memory':low=float(np.quantile(scores[yc==0],.99));high=max(low*1.5,float(np.nextafter(low,np.inf)))
        else:low,high=calibrate(yc,scores)
        thresholds[arm]=[low,high];calibration[arm]=measure(yc,scores,low,high);qualification[arm]=measure(yq,body_scores(xq,rq,params,arm),low,high)
        choices.append((calibration[arm]['balanced_accuracy'],-i,arm))
    selected=max(choices)[2];q=qualification[selected];qbody,_,qgap=scope_values(xq,params);qcover=float(np.mean((qbody>=minimum)&(qgap>=margin)))
    qualified=scope_fitted and q['recall']>=.9 and q['false_positive_rate']<=.1 and qcover>=.95
    save_arrays(output/'quality.npz',**params)
    metadata={'version':p['version'],'family_key':'screw','view':'m2_body','arrays':'quality.npz','arrays_sha256':sha(output/'quality.npz'),'protocol_sha256':sha(output/'protocol.json'),
        'weights_sha256':p['weights_sha256'],'selected_mode':selected,'thresholds':thresholds,'view_gate':{'min_body_similarity':minimum,'margin':margin},
        'qualified_for_research_trial':bool(qualified),'quality_label_basis':'publisher_OK_NG','automatic_archival':False,
        'scope':'M2 BODY publisher images only; research hints require review; no physical identity or production validation'}
    write(output/'quality.json',metadata)
    write(output/'development_report.json',{'selected_mode':selected,'calibration':calibration,'qualification':qualification,'view_scope_fitted':scope_fitted,
        'qualification_scope_coverage':qcover,'qualified_for_research_trial':bool(qualified),'test_read':False,'automatic_archival':False})
    print({'selected':selected,'qualified':bool(qualified),'qualification':q,'scope_coverage':qcover},flush=True)


def evaluate(output):
    if (output/'report.json').exists():raise ValueError('Completed BODY evaluation is frozen')
    p=read(output/'protocol.json');model=BodyQualityModel(output/'quality.json');params=model.params
    from perception.winclip_patch import WinCLIPPatch
    import torch
    torch.set_num_threads(2);detector=WinCLIPPatch(device='cpu');rows=[s for s in p['samples'] if s['role'] in ('test','head_control','layout_preview')];cache=output/'features';started=time.monotonic()
    missing=[s for s in rows if not (cache/(s['pixel_hash']+'.npz')).exists()]
    for start in range(0,len(missing),4):
        batch=missing[start:start+4];images=[];regions=[]
        for s in batch:
            if sha(ROOT/s['path'])!=s['file_sha256']:raise ValueError('Frozen evaluation source changed')
            with Image.open(ROOT/s['path']) as im:
                prepared,region=prepare_body(im);images.append(prepared);regions.append(region)
        patches,vectors=detector.extract_batch(images)
        for s,local,vector,region in zip(batch,patches,vectors,regions):save_arrays(cache/(s['pixel_hash']+'.npz'),vector=vector,raw=np.array(float(model.raw_scores(local).max())),roi_found=np.array(region['proposal_found']))
        if start%40==0:print({'sealed_features':start+len(batch),'total':len(missing),'seconds':round(time.monotonic()-started)},flush=True)
    body=[s for s in rows if s['role']=='test'];x,raw=load_features(output,body);y=np.array([s['label']=='defect' for s in body],dtype=int)
    metrics={arm:measure(y,body_scores(x,raw,params,arm),*model.meta['thresholds'][arm]) for arm in ARMS}
    results=[model.predict(vector,score) for vector,score in zip(x,raw)];keep=np.array([r['scope_ok'] for r in results])
    controls=[s for s in rows if s['role']=='head_control'];xh,rh=load_features(output,controls);head_results=[model.predict(vector,score) for vector,score in zip(xh,rh)]
    selected=model.meta['selected_mode'];scores=body_scores(x,raw,params,selected)
    report={'version':p['version'],'selected_mode':selected,'test_count':len(body),'test_normal':int((y==0).sum()),'test_defect':int((y==1).sum()),
        'methods':metrics,'scope_coverage':float(keep.mean()),'scope_rejected':int((~keep).sum()),
        'accepted_subset':measure(y[keep],scores[keep],*model.meta['thresholds'][selected]) if keep.any() else None,
        'head_controls':{'n':len(controls),'incorrectly_accepted_as_body':sum(r['scope_ok'] for r in head_results)},
        'development':read(output/'development_report.json'),'qualified_for_research_trial':model.meta['qualified_for_research_trial'],
        'arrays_sha256':model.meta['arrays_sha256'],'protocol_sha256':model.meta['protocol_sha256'],'quality_label_basis':'publisher_OK_NG',
        'automatic_archival':False,'scope':model.meta['scope'],'localization_evaluated':False}
    gate=p['sealed_trial_gate'];reasons=[];m=metrics[selected]
    if not model.meta['qualified_for_research_trial']:reasons.append('independent qualification gate failed')
    if m['recall']<gate['minimum_recall']:reasons.append('sealed test recall below floor')
    if m['false_positive_rate']>gate['maximum_false_positive_rate']:reasons.append('sealed test normal false-positive rate above ceiling')
    if report['scope_coverage']<gate['minimum_scope_coverage']:reasons.append('sealed BODY view coverage below floor')
    if 1-report['head_controls']['incorrectly_accepted_as_body']/len(controls)<gate['minimum_head_rejection']:reasons.append('HEAD rejection below floor')
    report['eligible_for_research_trial']=not reasons;report['trial_gate_reasons']=reasons
    official=[s for s in rows if s['role'] in ('test','layout_preview')];xo,ro=load_features(output,official);yo=np.array([s['label']=='defect' for s in official],dtype=int)
    report['official_test_including_layout_previews']={'n':len(official),'scope':'includes two pre-freeze layout inspection images; descriptive, not a sealed-test claim','selected_method':measure(yo,body_scores(xo,ro,params,selected),*model.meta['thresholds'][selected])}
    roi_found=0
    for s in body:
        with np.load(cache/(s['pixel_hash']+'.npz'),allow_pickle=False) as z:roi_found+=int(z['roi_found'])
    report['preprocessing_test']={'roi_proposal_found':roi_found,'full_image_fallback':len(body)-roi_found,'scope':'proposal count, not localization correctness; bounding-box ground truth unavailable'}
    write(output/'test_predictions.json',[{'pixel_hash':s['pixel_hash'],'truth':s['label'],**r} for s,r in zip(body,results)]);write(output/'report.json',report)
    print({'test':metrics,'scope_coverage':report['scope_coverage'],'head_controls':report['head_controls']},flush=True)


def demos(output):
    p=read(output/'protocol.json');selected=[]
    for label in ('normal','defect'):
        selected+=sorted((s for s in p['samples'] if s['role']=='test' and s['label']==label),key=lambda s:s['pixel_hash'])[:2]
    selected+=sorted((s for s in p['samples'] if s['role']=='head_control'),key=lambda s:s['pixel_hash'])[:2]
    images=output/'demo_images';images.mkdir(exist_ok=True);rows=[]
    for s in selected:
        path=images/(s['pixel_hash']+'.png')
        with Image.open(ROOT/s['path']) as image:image.convert('RGB').save(path)
        rows.append({'id':s['pixel_hash'],'file':path.relative_to(output).as_posix(),'file_sha256':sha(path),
            'source_view':s['view'],'source_label':s['label'],'source_partition':'sealed_test' if s['role']=='test' else 'historic_head_control',
            'selection':'first two by frozen pixel hash, not selected for correct predictions'})
    write(output/'demo_samples.json',{'dataset':p['dataset'],'source':'https://data.mendeley.com/datasets/2y6mcsyrpy/2','license':'CC BY 4.0','samples':rows})
    print({'demo_images':len(rows)},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','extract','fit','evaluate','demos']);parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    {'prepare':prepare,'extract':extract,'fit':fit,'evaluate':evaluate,'demos':demos}[args.action](args.output)


if __name__=='__main__':main()
