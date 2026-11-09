"""Frozen V2 experiment: expanded labelled training, photometric augmentation, fresh holdouts.

Historical V1 Test is regression only. No test cohort selects a model or threshold.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image,ImageEnhance

from experiments.m2_body_quality import ROOT,OUTPUT as PRIOR,save_arrays,load_features,measure,demos
from experiments.prepare_m2_body import sha
from perception.body_preprocess import prepare_body
from perception.body_quality import BodyQualityModel,body_scores
from perception.feature_backends import binary_probability,export_binary
from products.families import read,write
from review.workflow import measurement

OUTPUT=ROOT/'results/experiments/m2_body_quality_v2'
VARIANTS=('original','dim','bright')
CS=(.01,.1,1.)


def split_unused(rows,seed=20261107):
    result=[]
    for label in ('normal','defect'):
        group=sorted((s for s in rows if s['role']=='feedback' and s['label']==label and not s['excluded']),
            key=lambda s:hashlib.sha256((str(seed)+s['pixel_hash']).encode()).hexdigest())
        if len(group)<300:raise ValueError('Insufficient unused images for prespecified holdouts')
        for i,s in enumerate(group):
            result.append({**s,'path':s['source_path'],'view':'body','role':'test' if i<200 else 'qualification' if i<300 else 'train',
                'v1_role':'feedback','publisher_partition':s['source_partition']})
    return result


def prepare(output):
    source=ROOT/'data/external/m2_v2/body-manifest.json';audit=read(source);old=read(PRIOR/'protocol.json')
    samples=split_unused(audit['images'])
    for s in old['samples']:
        if s['role'] in ('train','calibration','qualification','head_control'):
            role='calibration' if s['role']=='qualification' else s['role']
        elif s['role']=='test':role='regression'
        else:continue
        samples.append({**s,'role':role,'v1_role':s['role'],'publisher_partition':s.get('source_partition')})
    groups={}
    for s in samples:groups.setdefault(s['pixel_hash'],set()).add(s['role'])
    if any(len(v)!=1 for v in groups.values()):raise ValueError('Image crossed V2 roles')
    p={'version':'m2-body-quality-v2-robust','dataset':audit['dataset'],'body_manifest_sha256':sha(source),
        'prior_protocol_sha256':sha(PRIOR/'protocol.json'),'prior_model_sha256':sha(PRIOR/'quality.npz'),
        'prior_report_sha256':sha(PRIOR/'report.json'),'samples':samples,'seed':20261107,
        'weights_sha256':old['weights_sha256'],'extractor_sha256':old['extractor_sha256'],'preprocessing_sha256':old['preprocessing_sha256'],
        'augmentation':'same fixed BODY crop; brightness multipliers 0.7 and 1.3; geometry unchanged; training only',
        'policies':['original','photometric'],'classifier_C':list(CS),'training_weight':'each original source image total weight 1; augmented images are not new independent samples',
        'calibration_target':{'minimum_recall':.93,'maximum_false_positive_rate':.08},
        'selection':'calibration target feasible first, then minimum false positive rate, maximum recall, strongest regularization, original policy on ties; no test selection',
        'qualification_gate':{'minimum_recall':.9,'maximum_false_positive_rate':.1,'minimum_scope_coverage':.95},
        'sealed_trial_gate':{'minimum_recall':.9,'maximum_false_positive_rate':.1,'minimum_scope_coverage':.95,'minimum_head_rejection':.95},
        'historical_regression_gate':{'minimum_recall':.9,'maximum_false_positive_rate':.1,'minimum_scope_coverage':.95},
        'view_gate':'reuse frozen V1 BODY/HEAD view gate and normal bank; no retraining from V1 Test',
        'fresh_test_source':'400 images from previously unused publisher Train, not a new factory or capture domain',
        'physical_identity':'not provided; image-level separation only','automatic_archival':False,'test_used_for_fitting':False}
    if (output/'protocol.json').exists() and read(output/'protocol.json')!=p:raise ValueError('Do not change frozen V2 protocol')
    write(output/'protocol.json',p)
    official=[s['pixel_hash'] for s in samples if s['role']=='test']
    write(output/'test_reservations.json',{'pixel_hashes':official,'primary_sealed_test_n':400,'layout_preview_n':0,
        'scope':'new project holdout from unused publisher Train; V1 Test remains regression and globally reserved'})
    reservation_path=ROOT/'results/evaluation_reservations.json';reserved=read(reservation_path)
    for key in official:reserved.setdefault('global',{}).setdefault(key,{'experiment':p['version'],'role':'test','dataset':'M2 BODY V2, publisher Train held out for V2'})
    write(reservation_path,reserved)
    print({'version':p['version'],'roles':dict(Counter(s['role'] for s in samples))},flush=True)


def augment(image,variant):
    if variant not in VARIANTS:raise ValueError('Unknown fixed augmentation')
    return image if variant=='original' else ImageEnhance.Brightness(image).enhance(.7 if variant=='dim' else 1.3)


def feature_path(output,sample,variant='original'):
    return output/'features'/(sample['pixel_hash']+('' if variant=='original' else '.'+variant)+'.npz')


def extract_rows(output,rows,*,training=False):
    p=read(output/'protocol.json')
    for path,key in [(ROOT/'weights/CLIP-ViT-B-16-open_clip_model.safetensors','weights_sha256'),(ROOT/'src/perception/winclip_patch.py','extractor_sha256'),(ROOT/'src/perception/body_preprocess.py','preprocessing_sha256')]:
        if sha(path)!=p[key]:raise ValueError('Frozen feature extractor changed')
    cache=output/'features';cache.mkdir(exist_ok=True);jobs=[]
    for s in rows:
        for variant in VARIANTS if training and s['role']=='train' else ('original',):
            target=feature_path(output,s,variant)
            if target.exists():continue
            old=PRIOR/'features'/(s['pixel_hash']+'.npz')
            if variant=='original' and old.exists():shutil.copy2(old,target)
            else:jobs.append((s,variant))
    if not jobs:return
    from perception.winclip_patch import WinCLIPPatch
    import torch
    torch.set_num_threads(2);detector=WinCLIPPatch(device='cpu');prior=BodyQualityModel(PRIOR/'quality.json');verified=set();started=time.monotonic()
    for start in range(0,len(jobs),8):
        batch=jobs[start:start+8];images=[];regions=[]
        for s,variant in batch:
            if s['pixel_hash'] not in verified:
                if sha(ROOT/s['path'])!=s['file_sha256']:raise ValueError('Frozen source image changed')
                verified.add(s['pixel_hash'])
            with Image.open(ROOT/s['path']) as image:prepared,region=prepare_body(image)
            images.append(augment(prepared,variant));regions.append(region)
        patches,vectors=detector.extract_batch(images)
        for (s,variant),local,vector,region in zip(batch,patches,vectors,regions):
            values={'vector':vector}
            if variant=='original':values.update(raw=np.array(float(prior.raw_scores(local).max())),roi_found=np.array(region['proposal_found']))
            save_arrays(feature_path(output,s,variant),**values)
        if start%80==0:print({'feature_jobs':start+len(batch),'total':len(jobs),'seconds':round(time.monotonic()-started),'training':training},flush=True)


def extract(output):
    p=read(output/'protocol.json');rows=[s for s in p['samples'] if s['role'] in ('train','calibration')]
    extract_rows(output,rows,training=True)


def threshold_candidates(scores):
    unique=np.unique(scores)
    return np.r_[np.nextafter(unique[0],-np.inf),(unique[:-1]+unique[1:])/2,np.nextafter(unique[-1],np.inf)]


def conservative_threshold(labels,scores,target):
    choices=[]
    for t in threshold_candidates(scores):
        m=measurement(labels,scores,t,np.inf);feasible=m['recall']>=target['minimum_recall'] and m['false_positive_rate']<=target['maximum_false_positive_rate']
        rank=(int(feasible),-m['false_positive_rate'],m['recall']) if feasible else (0,m['balanced_accuracy'],m['recall'])
        choices.append((rank,float(t),m))
    rank,low,m=max(choices,key=lambda row:row[0]);high=float(np.nextafter(max(float(max(scores)),low),np.inf))
    for t in threshold_candidates(scores):
        if t<=low:continue
        h=measurement(labels,scores,low,t)
        if h['high_risk_count']>=3 and h['high_risk_precision']>=.98 and h['high_risk_false_positives']/int(np.sum(np.asarray(labels)==0))<=.05:
            high=float(t);break
    return low,high,bool(rank[0])


def fit(output):
    if (output/'quality.json').exists():raise ValueError('Do not refit frozen V2 model')
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    p=read(output/'protocol.json');train=[s for s in p['samples'] if s['role']=='train'];cal=[s for s in p['samples'] if s['role']=='calibration']
    xc,rc=load_features(output,cal);yc=np.array([s['label']=='defect' for s in cal],int);y=np.array([s['label']=='defect' for s in train],int)
    params={};candidates={};choices=[]
    for policy in p['policies']:
        variants=VARIANTS if policy=='photometric' else ('original',);features=[]
        for variant in variants:
            vectors=[]
            for s in train:
                with np.load(feature_path(output,s,variant),allow_pickle=False) as z:vectors.append(z['vector'])
            features.append(np.asarray(vectors,dtype=float))
        x=np.concatenate(features);labels=np.tile(y,len(variants))
        for C in p['classifier_C']:
            name=f'{policy}_c{C:g}'.replace('.','p');pipeline=Pipeline([('sc',StandardScaler()),('clf',LogisticRegression(C=C,class_weight='balanced',max_iter=2000,random_state=p['seed']))])
            with threadpool_limits(limits=2):pipeline.fit(x,labels,clf__sample_weight=np.full(len(labels),1/len(variants)))
            portable=export_binary(pipeline,name);params.update(portable);scores=binary_probability(xc,portable,name)
            low,high,feasible=conservative_threshold(yc,scores,p['calibration_target']);m=measure(yc,scores,low,high)
            candidates[name]={'policy':policy,'C':C,'thresholds':[low,high],'target_feasible':feasible,'calibration':m}
            rank=(int(feasible),-m['false_positive_rate'],m['recall'],-C,int(policy=='original')) if feasible else (0,m['balanced_accuracy'],m['recall'],-C,int(policy=='original'))
            choices.append((rank,name))
    selected=max(choices)[1];chosen=candidates[selected];prior=BodyQualityModel(PRIOR/'quality.json')
    model_params={k:prior.params[k].copy() for k in ('bank','body_centroid','head_centroid')}
    model_params.update({key.replace(selected+'_','global_',1):value for key,value in params.items() if key.startswith(selected+'_')})
    save_arrays(output/'candidate_models.npz',**params);save_arrays(output/'quality.npz',**model_params)
    metadata={**prior.meta,'version':p['version'],'arrays_sha256':sha(output/'quality.npz'),'protocol_sha256':sha(output/'protocol.json'),
        'selected_mode':'global_binary','thresholds':{'global_binary':chosen['thresholds']},'qualified_for_research_trial':False,
        'training':{'source_images':len(train),'calibration_images':len(cal),'variant':selected,'policy':chosen['policy'],'C':chosen['C'],'source_label_basis':'publisher OK/NG'},
        'training_implementation_sha256':sha(Path(__file__)),
        'scope':'M2 BODY research candidate; fresh holdouts are unused publisher Train, historical Test is regression only; no production archival'}
    write(output/'quality.json',metadata)
    write(output/'development_report.json',{'selected_candidate':selected,'candidates':candidates,'qualification_read':False,'test_read':False,'regression_read':False,
        'candidate_arrays_sha256':sha(output/'candidate_models.npz'),'training_source_images':len(train),'calibration_source_images':len(cal),'automatic_archival':False})
    print({'selected':selected,'calibration':chosen,'qualification_not_read':True,'test_not_read':True},flush=True)


def cohort(output,rows,model):
    x,raw=load_features(output,rows);y=np.array([s['label']=='defect' for s in rows],int)
    score=body_scores(x,raw,model.params,model.meta['selected_mode']);pred=[model.predict(vector,value) for vector,value in zip(x,raw)]
    m=measure(y,score,*model.meta['thresholds'][model.meta['selected_mode']]);coverage=float(np.mean([s['scope_ok'] for s in pred]))
    return m,coverage,pred,x,raw,y


def failed_gate(metrics,coverage,gate):
    return metrics['recall']<gate['minimum_recall'] or metrics['false_positive_rate']>gate['maximum_false_positive_rate'] or coverage<gate['minimum_scope_coverage']


def qualify(output):
    if (output/'qualification_report.json').exists() or (output/'report.json').exists():raise ValueError('Qualification is frozen')
    p=read(output/'protocol.json');rows=[s for s in p['samples'] if s['role']=='qualification'];extract_rows(output,rows)
    model=BodyQualityModel(output/'quality.json');m,coverage,*_=cohort(output,rows,model);passed=not failed_gate(m,coverage,p['qualification_gate'])
    write(output/'qualification_report.json',{'metrics':m,'scope_coverage':coverage,'passed':passed,'source':'previously unused publisher Train holdout','used_for_fitting':False})
    meta=read(output/'quality.json');meta['qualified_for_research_trial']=bool(passed);write(output/'quality.json',meta)
    print({'qualification':m,'scope_coverage':coverage,'passed':passed},flush=True)


def evaluate(output):
    if (output/'report.json').exists():raise ValueError('Final evaluation is frozen')
    p=read(output/'protocol.json');qualification=read(output/'qualification_report.json');rows=[s for s in p['samples'] if s['role'] in ('test','regression','head_control')];extract_rows(output,rows)
    model=BodyQualityModel(output/'quality.json');prior=BodyQualityModel(PRIOR/'quality.json');test=[s for s in rows if s['role']=='test']
    m,coverage,pred,x,raw,y=cohort(output,test,model);base=measure(y,body_scores(x,raw,prior.params,prior.meta['selected_mode']),*prior.meta['thresholds'][prior.meta['selected_mode']])
    regression=[s for s in rows if s['role']=='regression'];mr,cr,*_=cohort(output,regression,model)
    controls=[s for s in rows if s['role']=='head_control'];xh,rh=load_features(output,controls);incorrect=sum(model.predict(vector,value)['scope_ok'] for vector,value in zip(xh,rh))
    reasons=[]
    if not qualification['passed']:reasons.append('independent qualification gate failed')
    gate=p['sealed_trial_gate']
    if m['recall']<gate['minimum_recall']:reasons.append('sealed test recall below floor')
    if m['false_positive_rate']>gate['maximum_false_positive_rate']:reasons.append('sealed test normal false-positive rate above ceiling')
    if coverage<gate['minimum_scope_coverage']:reasons.append('sealed BODY view coverage below floor')
    if 1-incorrect/len(controls)<gate['minimum_head_rejection']:reasons.append('HEAD rejection below floor')
    if failed_gate(mr,cr,p['historical_regression_gate']):reasons.append('historical regression gate failed')
    policy=model.meta['training']['policy'];display='扩充训练＋明暗增强' if policy=='photometric' else '扩充训练整图判别'
    report={'version':p['version'],'selected_mode':'global_binary','selected_candidate':read(output/'development_report.json')['selected_candidate'],
        'method_names':{'v1_fixed':'V1冻结融合方案','global_binary':display},'method_note':'两方案均使用来源质量标签；新方案增加训练样本，候选及阈值只按校准集选择。冻结正常库及视图检查保持一致。',
        'test_partition_description':'新增400张此前未用的发布方Train图片，按图像封存；不代表新的厂家或采集域。原998张Test仅作历史回归。','test_count':len(test),'test_normal':int((y==0).sum()),'test_defect':int((y==1).sum()),
        'methods':{'v1_fixed':base,'global_binary':m},'scope_coverage':coverage,'scope_rejected':sum(not item['scope_ok'] for item in pred),
        'head_controls':{'n':len(controls),'incorrectly_accepted_as_body':int(incorrect)},'qualification':qualification,
        'historical_regression':{'n':len(regression),'current':mr,'baseline':read(PRIOR/'report.json')['methods']['fast_fusion'],'scope_coverage':cr,'not_new_blind':True},
        'previous_report':read(PRIOR/'report.json'),'arrays_sha256':model.meta['arrays_sha256'],'protocol_sha256':model.meta['protocol_sha256'],
        'quality_label_basis':'publisher_OK_NG','automatic_archival':False,'scope':model.meta['scope'],'localization_evaluated':False,
        'eligible_for_research_trial':not reasons,'trial_gate_reasons':reasons,'test_used_for_fitting':False,'regression_used_for_selection':False}
    write(output/'test_predictions.json',[{'pixel_hash':s['pixel_hash'],'truth':s['label'],**result} for s,result in zip(test,pred)])
    write(output/'report.json',report)
    print({'fresh':m,'baseline_fresh':base,'historical_regression':mr,'eligible':not reasons,'reasons':reasons},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','extract','fit','qualify','evaluate','demos']);parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    {'prepare':prepare,'extract':extract,'fit':fit,'qualify':qualify,'evaluate':evaluate,'demos':demos}[args.action](args.output)


if __name__=='__main__':main()
