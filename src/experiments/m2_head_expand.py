"""Single labelled HEAD training expansion, with frozen historical veto checks."""
import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image

from experiments.m2_body_kernel import PROTECTED, ROOT
from experiments.m2_body_quality import save_arrays, measure
from experiments.prepare_m2_body import sha
from perception.feature_backends import export_binary, binary_probability
from products.families import read, write
from review.workflow import calibrate, image_hash

KEY = 'product_a269405b7b9a'
BUILD = ROOT/'results/product_catalog'/KEY/'builds/13ff4b2995df4239a4d16d133c48ef6b'
OUTPUT = ROOT/'results/experiments/m2_head_expanded_v1'


def select_training(samples, reservations):
    blocked = set(reservations.get('global', {})) | set(reservations.get('products', {}).get(KEY, {}))
    rows = [{**s,'source_role':s['role'],'role':'train'} for s in samples
            if s['role'] in ('train','feedback') and s['pixel_hash'] not in blocked]
    if len({s['pixel_hash'] for s in rows}) != len(rows): raise ValueError('Duplicate training image')
    return rows


def prepare(output):
    if (output/'protocol.json').exists(): raise ValueError('Protocol frozen')
    source = read(BUILD/'input.json'); reservations = read(ROOT/'results/evaluation_reservations.json')
    training = select_training(source['samples'], reservations)
    others = [s for s in source['samples'] if s['role'] in ('calibration','qualification','test')]
    # Existing family illumination Test must not be silently reused as training.
    family = read(ROOT/'results/experiments/screw_family_v2/protocol.json')
    by_hash = {s['pixel_hash']:s for s in source['samples']}
    family_test = [{**by_hash[s['pixel_hash']], 'role':'family_regression'} for s in family['samples']
                   if s['group'] == 1 and s['role'] == 'new_test']
    rows = training + others + family_test
    if len({s['pixel_hash'] for s in rows}) != len(rows): raise ValueError('Image crossed roles')
    config = next(c for c in read(ROOT/'results/perception/banks/products.json') if c['key'] == KEY)
    meta_path = ROOT/'results/perception/optimized'/config['perception_artifact']
    meta = read(meta_path)
    sources = [BUILD/'input.json', BUILD/(KEY+'_features.npz'), meta_path, meta_path.parent/meta['arrays'],
               ROOT/'weights/CLIP-ViT-B-16-open_clip_model.safetensors', ROOT/'src/perception/winclip_patch.py']
    p = {'version':'m2-head-expanded-v1', 'samples':rows, 'base_config':config, 'C':.1,
         'implementation_sha256':sha(Path(__file__)),
         'sources':{str(path.relative_to(ROOT)):sha(path) for path in sources},
         'protected':{name:sha(ROOT/name) for name in PROTECTED},
         'candidate_count':1, 'change':'add nonreserved publisher-labelled Train images; same encoder and linear classifier C=.1',
         'gate':{'minimum_recall':.9,'maximum_false_positive_rate':.1},
         'selection':'no hyperparameter search; original calibration-only balanced-accuracy threshold rule',
         'stop_rule':'stop if calibration quality regresses; otherwise replay historical cohorts once; require no quality regression',
         'scope':'all evaluation cohorts are historical public-data regression, not new blind validation',
         'release_permission':False, 'automatic_archival':False}
    output.mkdir(parents=True,exist_ok=True);write(output/'protocol.json',p)
    print({'training':len(training),'additional_training':sum(s['source_role']=='feedback' for s in training),
           'evaluation_roles':{role:sum(s['role']==role for s in rows) for role in ('calibration','qualification','test','family_regression')}},flush=True)


def verify(output):
    p = read(output/'protocol.json')
    if sha(Path(__file__)) != p['implementation_sha256']: raise ValueError('Implementation changed')
    if any(sha(ROOT/name) != h for name,h in {**p['sources'],**p['protected']}.items()): raise ValueError('Protected source/state changed')
    return p


def extract(output):
    p = verify(output)
    from experiments.screw_family import Features
    family = read(ROOT/'results/experiments/screw_family_v2/protocol.json')
    cache = Features(ROOT/'results/experiments/screw_family_v2',family)
    wanted = [s for s in p['samples'] if s['role'] in ('train','calibration')]
    vectors = {}; missing = []
    with np.load(BUILD/(KEY+'_features.npz'),allow_pickle=False) as z:
        known = {str(path):x.copy() for path,x in zip(z['paths'],z['global_features'])}
    for s in wanted:
        destination = output/'features'/(s['pixel_hash']+'.npz')
        if destination.exists(): continue
        if s['path'] in known: vectors[s['pixel_hash']] = known[s['path']]
        elif cache.path(s,'original').exists(): vectors[s['pixel_hash']] = cache.get(s)[1]
        else: missing.append(s)
    (output/'features').mkdir(exist_ok=True)
    for fingerprint,x in vectors.items(): save_arrays(output/'features'/(fingerprint+'.npz'),vector=x)
    if not missing: return
    import torch
    from perception.winclip_patch import WinCLIPPatch
    torch.set_num_threads(2); detector = WinCLIPPatch(device='cpu'); started=time.monotonic()
    for start in range(0,len(missing),8):
        batch = missing[start:start+8]; images=[]
        for s in batch:
            path=ROOT/s['path']
            if image_hash(path)!=s['pixel_hash']:raise ValueError('Source image changed')
            with Image.open(path) as image:images.append(image.convert('RGB'))
        _,values = detector.extract_batch(images)
        for s,x in zip(batch,values):save_arrays(output/'features'/(s['pixel_hash']+'.npz'),vector=x)
        if start%80==0 or start+len(batch)==len(missing):print({'extracted':start+len(batch),'total':len(missing),'seconds':round(time.monotonic()-started)},flush=True)


def arrays(output,rows):
    values=[]
    for s in rows:
        with np.load(output/'features'/(s['pixel_hash']+'.npz'),allow_pickle=False) as z:values.append(z['vector'])
    return np.asarray(values,dtype=float),np.array([s['label']=='defect' for s in rows],int)


def no_regression(old,new):
    return all(new[k]+1e-12>=old[k] for k in ('accuracy','balanced_accuracy','recall')) and new['false_positive_rate']<=old['false_positive_rate']+1e-12


def fit(output):
    if (output/'candidate.json').exists():raise ValueError('Candidate frozen')
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    p=verify(output); train=[s for s in p['samples'] if s['role']=='train']; cal=[s for s in p['samples'] if s['role']=='calibration']
    x,y=arrays(output,train);xc,yc=arrays(output,cal)
    model=Pipeline([('sc',StandardScaler()),('clf',LogisticRegression(C=p['C'],class_weight='balanced',max_iter=2000,random_state=20261108))])
    with threadpool_limits(limits=2):model.fit(x,y)
    params=export_binary(model,'global');score=binary_probability(xc,params,'global');low,high=calibrate(yc,score)
    baseline=read(BUILD/'report.json')['calibration'];metric=measure(yc,score,low,high)
    save_arrays(output/'candidate.npz',**params)
    write(output/'candidate.json',{'arrays_sha256':sha(output/'candidate.npz'),'protocol_sha256':sha(output/'protocol.json'),
          'thresholds':[low,high],'calibration':metric,'base_calibration':baseline,'development_passed':no_regression(baseline,metric),
          'release_permission':False,'automatic_archival':False})
    print({'calibration':metric,'no_regression':no_regression(baseline,metric)},flush=True)


def evaluate(output):
    if (output/'report.json').exists():raise ValueError('Evaluation frozen')
    p=verify(output);meta=read(output/'candidate.json')
    if sha(output/'candidate.npz')!=meta['arrays_sha256'] or sha(output/'protocol.json')!=meta['protocol_sha256']:raise ValueError('Unpaired candidate')
    with np.load(output/'candidate.npz',allow_pickle=False) as z:params={k:z[k].copy() for k in z.files}
    report={'version':p['version'],'scope':p['scope'],'release_permission':False,'automatic_archival':False,'cohorts':{},'stop_reason':None}
    if not meta['development_passed']:report['stop_reason']='calibration quality regressed; heldout features not read'
    else:
        from experiments.screw_family import Features
        cache=Features(ROOT/'results/experiments/screw_family_v2',read(ROOT/'results/experiments/screw_family_v2/protocol.json'))
        with np.load(BUILD/(KEY+'_features.npz'),allow_pickle=False) as z:known={str(path):v.copy() for path,v in zip(z['paths'],z['global_features'])}
        config=p['base_config'];baseline_meta=read(ROOT/'results/perception/optimized'/config['perception_artifact'])
        with np.load(ROOT/'results/perception/optimized'/baseline_meta['arrays'],allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
        predictions=[]
        for role in ('qualification','test','family_regression'):
            rows=[s for s in p['samples'] if s['role']==role]
            x=np.asarray([known[s['path']] if s['path'] in known else cache.get(s)[1] for s in rows],dtype=float)
            y=np.array([s['label']=='defect' for s in rows],int);score=binary_probability(x,params,'global');old=binary_probability(x,base,'global')
            current=measure(y,old,config['threshold'],config['t_high']);new=measure(y,score,*meta['thresholds'])
            report['cohorts'][role]={'current':current,'candidate':new,'no_regression':no_regression(current,new)}
            predictions.extend({'pixel_hash':s['pixel_hash'],'role':role,'truth':s['label'],'current_score':float(a),'candidate_score':float(b)} for s,a,b in zip(rows,old,score))
            print({role:report['cohorts'][role]},flush=True)
            if not no_regression(current,new) or new['recall']<.9 or new['false_positive_rate']>.1:
                report['stop_reason']=role+' failed quality gate; stop without further tuning';break
        write(output/'predictions.json',predictions)
    if not report['stop_reason']:report['stop_reason']='historical gates passed; independent validation and automatic disposition qualification remain outstanding'
    verify(output);write(output/'report.json',report);print({'stop_reason':report['stop_reason'],'production_permission':False},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','extract','fit','evaluate']);parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();{'prepare':prepare,'extract':extract,'fit':fit,'evaluate':evaluate}[args.action](args.output.resolve())


if __name__=='__main__':main()
