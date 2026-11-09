"""Bundle/install the verified HEAD scorer, retaining mandatory human review."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

KEY='product_a269405b7b9a'
VERSION='head_expanded_20261108'
DIRECTORY='experiments/m2_head_expanded_v1/review_benchmark'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    prior=p.stat() if p.exists() else None
    temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if prior is not None and os.name=='posix':
        temp.chmod(prior.st_mode & 0o777)
        if os.geteuid()==0:os.chown(temp,prior.st_uid,prior.st_gid)
    temp.replace(p)


def validate_gate(report, candidate):
    if candidate['release_permission'] or report['release_permission']:raise ValueError('Research must not grant automatic permission')
    groups=[{'current':candidate['base_calibration'],'candidate':candidate['calibration']}]
    if set(report['cohorts'])!={'qualification','test','family_regression'}:raise ValueError('Missing historical cohort')
    groups+=list(report['cohorts'].values())
    for group in groups:
        old,new=group['current'],group['candidate']
        for m in (old,new):
            (tn,fp),(fn,tp)=m['confusion_matrix'];n=tn+fp+fn+tp
            if n!=m['n'] or min(tn,fp,fn,tp)<0:raise ValueError('Invalid confusion counts')
            checks={'accuracy':(tn+tp)/n,'recall':tp/(tp+fn),'false_positive_rate':fp/(tn+fp)}
            if any(abs(m[k]-v)>1e-10 for k,v in checks.items()):raise ValueError('Inconsistent metrics')
        if any(new[k]+1e-12<old[k] for k in ('accuracy','balanced_accuracy','recall')) or new['false_positive_rate']>old['false_positive_rate']+1e-12:
            raise ValueError('Quality regression')
        if new['recall']<.9 or new['false_positive_rate']>.1:raise ValueError('Quality floor failed')


def build(root,stage):
    import numpy as np
    results=root/'results';research=results/'experiments/m2_head_expanded_v1'
    report=read(research/'report.json');candidate=read(research/'candidate.json');p=read(research/'protocol.json')
    validate_gate(report,candidate)
    if stage.exists():raise ValueError('Stage already exists')
    if sha(research/'candidate.npz')!=candidate['arrays_sha256'] or sha(research/'protocol.json')!=candidate['protocol_sha256']:raise ValueError('Candidate changed')
    configs=read(results/'perception/banks/products.json');old=next(c for c in configs if c['key']==KEY)
    if old!=p['base_config']:raise ValueError('Active HEAD differs from tested baseline')
    old_build=results/old['review_benchmark'];source=read(old_build/'input.json')
    training=[s for s in p['samples'] if s['role']=='train'];cal=[s for s in p['samples'] if s['role']=='calibration'];qual=[s for s in p['samples'] if s['role']=='qualification']
    bank=[s for s in source['samples'] if s['role']=='bank'];rows=training+cal+qual+bank
    if len({s['pixel_hash'] for s in rows})!=len(rows):raise ValueError('Benchmark image crossed partitions')
    reservations=read(results/'evaluation_reservations.json')
    blocked=set(reservations.get('global',{}))|set(reservations.get('products',{}).get(KEY,{}))
    if any(s['pixel_hash'] in blocked for s in training):raise ValueError('Evaluation image entered training')
    split={'train':[s['path'] for s in training],'validation':[s['path'] for s in cal],
           'normal_calibration':[],'normal_memory':[s['path'] for s in bank],'test':[s['path'] for s in qual]}
    with np.load(old_build/(KEY+'_features.npz'),allow_pickle=False) as z:
        original={str(path):(v.copy(),float(raw)) for path,v,raw in zip(z['paths'],z['global_features'],z['clip_scores'])}
        normals=z['normal_global'].copy()
    features=[];raw=[];known=[]
    for s in rows:
        if s['path'] in original:v,r=original[s['path']];available=True
        else:
            with np.load(research/'features'/(s['pixel_hash']+'.npz'),allow_pickle=False) as z:v=z['vector'].copy()
            r=float('nan');available=False
        features.append(v);raw.append(r);known.append(available)
    # New source rows have global features only; missing patch scores are explicit,
    # never fabricated. This benchmark is bound to global_binary mode.
    folder=stage/'results'/DIRECTORY;folder.mkdir(parents=True)
    with (folder/(KEY+'_features.npz')).open('wb') as stream:
        np.savez_compressed(stream,signature=np.array(hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest()),
             paths=np.array([s['path'] for s in rows]),global_features=np.asarray(features),labels=np.array([s['label']=='defect' for s in rows],int),
             normal_global=normals,clip_scores=np.array(raw),clip_score_available=np.array(known))
    write(folder/(KEY+'_split.json'),split)
    training_hashes={s['pixel_hash'] for s in training};test_hashes={s['pixel_hash'] for s in p['samples'] if s['role'] in ('test','family_regression')}
    manifest=copy.deepcopy(source)
    for s in manifest['samples']:
        if s['pixel_hash'] in training_hashes:s['source_role']=s['role'];s['role']='train'
        if s['pixel_hash'] in test_hashes:s['source_role']=s.get('source_role',s['role']);s['role']='test'
    manifest['quality_label_basis']='publisher_OK_NG';write(folder/'input.json',manifest)
    write(folder/'provenance.json',{'mode':'global_binary','training_n':len(training),'normal_bank_unchanged':True,
         'missing_patch_scores':int(len(rows)-sum(known)),'qualification_scope':'historical_only; automatic action remains paused'})
    optimized=stage/'results/perception/optimized';optimized.mkdir(parents=True)
    artifact=VERSION+'.json';shutil.copy2(research/'candidate.npz',optimized/(VERSION+'.npz'))
    low,high=candidate['thresholds']
    metadata={'product':KEY,'mode':'global_binary','name':'CLIP 有标签整图判别 · 扩充训练',
              'arrays':VERSION+'.npz','threshold':low,'t_high':high,'review_version':VERSION,
              'source_training_n':len(training),'automatic_archival':False,'label_basis':'publisher_OK_NG'}
    write(optimized/artifact,metadata)
    config={**old,'threshold':low,'t_high':high,'perception_artifact':artifact,'perception_name':metadata['name'],
            'review_version':VERSION,'review_benchmark':DIRECTORY,'catalog_manifest':DIRECTORY+'/input.json'}
    write(stage/'report.json',report);write(stage/'candidate.json',candidate)
    files={path.relative_to(stage).as_posix():sha(path) for path in (stage/'results').rglob('*') if path.is_file()}
    write(stage/'release.json',{'version':VERSION,'product':KEY,'baseline_config':old,'config':config,'files':files,
          'report_sha256':sha(stage/'report.json'),'candidate_sha256':sha(stage/'candidate.json'),
          'scope':'assisted scoring update only; mandatory human review; historical validation',
          'automatic_archival':False})
    print(json.dumps({'built':str(stage),'files':len(files),'training':len(training),'mandatory_review':True}),flush=True)


def validate_stage(stage):
    bundle=read(stage/'release.json')
    if bundle['version']!=VERSION or bundle['product']!=KEY or bundle['automatic_archival'] is not False:raise ValueError('Invalid release scope')
    if sha(stage/'report.json')!=bundle['report_sha256'] or sha(stage/'candidate.json')!=bundle['candidate_sha256']:raise ValueError('Evidence changed')
    validate_gate(read(stage/'report.json'),read(stage/'candidate.json'))
    for name,expected in bundle['files'].items():
        path=(stage/name).resolve()
        if not path.is_relative_to((stage/'results').resolve()) or sha(path)!=expected:raise ValueError('Invalid package member')
    config=bundle['config'];meta=read(stage/'results/perception/optimized'/config['perception_artifact'])
    if config['key']!=KEY or meta['product']!=KEY or meta['mode']!='global_binary' or meta['automatic_archival'] is not False:raise ValueError('Invalid model binding')
    if config['threshold']!=meta['threshold'] or config['t_high']!=meta['t_high']:raise ValueError('Threshold mismatch')
    return bundle


def install(root,stage,backup):
    bundle=validate_stage(stage);results=root/'results';registry=results/'perception/banks/products.json';policy_path=results/'inspection_policy.json'
    configs=read(registry);old=next(c for c in configs if c['key']==KEY)
    if any(read(p).get('status') in ('queued','running','importing') for p in (results/'batches').glob('*/state.json')):
        raise ValueError('Active batch; defer model switch')
    if old!=bundle['baseline_config']:raise ValueError('HEAD no longer matches baseline; do not overwrite')
    if sha(results/'perception/banks'/old['bank'])!=old['bank_sha256']:raise ValueError('Normal bank changed')
    if backup.exists():raise ValueError('Backup already exists')
    backup.mkdir(parents=True);shutil.copy2(registry,backup/'products.json');shutil.copy2(policy_path,backup/'inspection_policy.json')
    for name in bundle['files']:
        source=stage/name;target=root/name
        if target.exists() and sha(target)!=sha(source):raise ValueError('Cannot overwrite frozen artifact')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(source,target)
    config=bundle['config'];meta=results/'perception/optimized'/config['perception_artifact'];model=meta.parent/read(meta)['arrays']
    model_key=hashlib.sha256((json.dumps(config,sort_keys=True)+sha(meta)+sha(model)).encode()).hexdigest()
    hold_key=KEY+':'+model_key;policy=read(policy_path)
    hold={'product':KEY,'model_key':model_key,'source':'assisted_head_update','release':VERSION,
          'automatic_release_vetoed':True,'reason':'扩充训练候选仅更新辅助评分，独立自动处置资格尚未通过；继续人工复核'}
    policy.setdefault('holds',{})[hold_key]=hold;policy['revision']=VERSION
    # Install the review hold first: a process can never see the new config without it.
    write(policy_path,policy)
    write(registry,[config if c['key']==KEY else c for c in configs])
    write(backup/'receipt.json',{'version':VERSION,'config':config,'hold_key':hold_key,'hold':hold,'rollback_available':True})
    print(json.dumps({'installed':VERSION,'mandatory_review':True,'hold_key':hold_key,'backup':str(backup)},ensure_ascii=False),flush=True)


def rollback(root,backup):
    receipt=read(backup/'receipt.json');results=root/'results';registry=results/'perception/banks/products.json'
    configs=read(registry);current=next(c for c in configs if c['key']==KEY)
    if current!=receipt['config']:raise ValueError('Active model changed since release; refuse rollback overwrite')
    old=next(c for c in read(backup/'products.json') if c['key']==KEY)
    write(registry,[old if c['key']==KEY else c for c in configs])
    # Keep the new version hold as well; an old reference must never unpause it.
    print(json.dumps({'rolled_back':VERSION,'holds_preserved':True}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['build','install','rollback'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--stage',type=Path);parser.add_argument('--backup',type=Path)
    a=parser.parse_args();root=a.root.resolve()
    if a.action=='build':build(root,a.stage.resolve())
    elif a.action=='install':install(root,a.stage.resolve(),a.backup.resolve())
    else:rollback(root,a.backup.resolve())


if __name__=='__main__':main()
