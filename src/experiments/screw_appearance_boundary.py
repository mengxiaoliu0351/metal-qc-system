"""Frozen V2 exclusion experiment with a newly reserved source-frame holdout."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import shutil

import numpy as np

from experiments.screw_appearance import extract,stats,sha
from perception.appearance import AppearanceMatcher,ranked
from perception.appearance_boundary import BoundaryAppearanceMatcher,boundary_ranked
from products.families import read,write

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'results/experiments/screw_appearance_v1'


def reserve(results,values):
    path=results/'product_families/screw/appearance_reservations.json'
    state=read(path) if path.exists() else {'schema':1,'family_key':'screw','samples':{}}
    for key,value in values['samples'].items():
        if key in state['samples'] and state['samples'][key]!=value:raise ValueError('Conflicting frozen appearance reservation')
    state['samples'].update(values['samples']);write(path,state)


def prepare(output):
    output.mkdir(parents=True,exist_ok=True);path=output/'protocol.json'
    if path.exists():
        p=read(path)
        if p['manifest_sha256']!=sha(ROOT/'results/product_families/screw/appearance.json'):raise ValueError('Source manifest changed')
        reserve(ROOT/'results',read(output/'fresh_reservations.json'));return
    old=read(OLD/'protocol.json');snapshot=read(OLD/'review_snapshot.json');manifest=read(ROOT/'results/product_families/screw/appearance.json')
    reviewed={s['key'] for s in snapshot['reviews']};seen={s['source_frame'] for s in manifest['samples'] if s['key'] in reviewed}
    previous=ROOT/'results/product_families/screw/appearance_reservations.json'
    reserved_frames={s['source_frame'] for s in read(previous)['samples'].values()} if previous.exists() else set()
    pool=[s for s in manifest['samples'] if s['split']=='train' and s['source_frame'] not in seen|reserved_frames]
    frames=sorted({s['source_frame'] for s in pool},key=lambda x:hashlib.sha256(('appearance-v2-fresh:'+x).encode()).hexdigest())[:40]
    if len(frames)!=40:raise ValueError('Need 40 unused source frames')
    fresh=[s for s in pool if s['source_frame'] in frames];categories=[t['category_id'] for t in old['reference_categories']]
    excluded=[t for t in manifest['types'] if t['category_id'] not in categories];negatives=[]
    for category in excluded:
        candidates=sorted([s for s in pool if s['source_frame'] not in frames and s['category_id']==category['category_id']],
            key=lambda s:hashlib.sha256(('appearance-v2-exclusion:'+s['key']).encode()).hexdigest())
        selected=[];used=set()
        for s in candidates:
            if s['source_frame'] in used:continue
            selected.append(s);used.add(s['source_frame'])
            if len(selected)==12:break
        if len(selected)!=12:raise ValueError('Insufficient exclusion frame diversity')
        negatives.extend(selected)
    rows=[{**s,'role':'regression' if s['role']=='test' else s['role']} for s in old['samples']]
    rows.extend({**s,'path':'results/'+s['path'],'role':'exclusion'} for s in negatives)
    rows.extend({**s,'path':'results/'+s['path'],'role':'fresh_test'} for s in fresh)
    for field in ('source_frame','pixel_hash'):
        partitions={}
        for s in rows:
            if field not in s:continue
            partitions.setdefault(s[field],set()).add(s['role'])
        if any(len(parts)>1 for parts in partitions.values()):raise ValueError('Cross-partition appearance leak')
    p={**old,'version':'screw-appearance-v2','samples':rows,'matcher_kind':'source-exclusion-v1','methods':['clip'],
        'reference_counts':old['reference_counts'],'exclusion_categories':excluded,'exclusion_reference_count':len(negatives),
        'exclusion_provenance':'publisher object category labels, not human quality labels or normal references',
        'baseline_arrays_sha256':read(OLD/'appearance.json')['arrays_sha256'],'baseline_protocol_sha256':sha(OLD/'protocol.json'),
        'fresh_frame_count':40,'fresh_sample_count':len(fresh),'fresh_source':'unused original training frames, excluding all prior reviewed and evaluation frames',
        'selection':'validation only: .5 known_correct_acceptance + .5 other_rejection; ties prefer known coverage',
        'gate_grid':{'score_quantiles':21,'exclusion_margin':[-.04,-.02,-.01,0,.005,.01,.02,.04,.08]},
        'promotion':{'max_known_correct_acceptance_drop':.02,'min_fresh_known_coverage':.8,
            'require_fresh_unknown_rate_reduction':True,'require_precision_non_decrease':True,'no_more_accepted_known_errors':True,
            'require_regression_unknown_rate_reduction':True,'no_more_control_false_accepts':True},
        'scope':'category retrieval only; new original-frame holdout within existing public dataset; old 473 are development regression, not fresh blind test; physical identities unknown'}
    reservations={'schema':1,'family_key':'screw','samples':{s['key']:{'experiment':'screw_appearance_v2','purpose':'fresh_appearance_evaluation',
        'source_frame':s['source_frame'],'pixel_hash':s['pixel_hash']} for s in fresh}}
    shutil.copy2(OLD/'review_snapshot.json',output/'review_snapshot.json')
    write(output/'fresh_reservations.json',reservations);write(path,p);reserve(ROOT/'results',reservations)
    print({'reference':24,'exclusion':len(negatives),'fresh_frames':40,'fresh_images':len(fresh)},flush=True)


def features(output):
    p=read(output/'protocol.json');cache=output/'features';cache.mkdir(exist_ok=True)
    for s in p['samples']:
        source=OLD/'features'/(s['key']+'.npz');target=cache/source.name
        if source.exists() and not target.exists():shutil.copy2(source,target)
    extract(output)


def prepare_conservative(output):
    """A new version and new frames after V2 failed its unchanged release gate."""
    output.mkdir(parents=True,exist_ok=True);target=output/'protocol.json'
    if target.exists():
        reserve(ROOT/'results',read(output/'fresh_reservations.json'));return
    previous=ROOT/'results/experiments/screw_appearance_v2';p=read(previous/'protocol.json')
    if read(previous/'report.json')['promotion']['eligible']:raise ValueError('This conservative branch records a failed V2, not silent retuning')
    manifest=read(ROOT/'results/product_families/screw/appearance.json');used={s['source_frame'] for s in p['samples']}
    existing=read(ROOT/'results/product_families/screw/appearance_reservations.json')
    used.update(s['source_frame'] for s in existing['samples'].values())
    candidates=[s for s in manifest['samples'] if s['split']=='train' and s['source_frame'] not in used]
    frames=sorted({s['source_frame'] for s in candidates},key=lambda x:hashlib.sha256(('appearance-v3-fresh:'+x).encode()).hexdigest())[:40]
    if len(frames)!=40:raise ValueError('Need 40 genuinely unused source frames')
    fresh=[s for s in candidates if s['source_frame'] in frames]
    rows=[{**s,'role':'previous_fresh_regression' if s['role']=='fresh_test' else s['role']} for s in p['samples']]
    rows.extend({**s,'path':'results/'+s['path'],'role':'fresh_test'} for s in fresh)
    for field in ('source_frame','pixel_hash'):
        groups={}
        for s in rows:
            if field in s:groups.setdefault(s[field],set()).add(s['role'])
        if any(len(parts)>1 for parts in groups.values()):raise ValueError('Conservative holdout overlaps development data')
    baseline=read(OLD/'appearance.json')
    p.update(version='screw-appearance-v3',samples=rows,fresh_sample_count=len(fresh),
        fixed_positive_gate=baseline['gate'],calibration_max_correct_acceptance_drop=.02,
        selection='keep V1 positive gate fixed; choose exclusion gap on validation only with <=2pp known correct acceptance loss; never widen V1 acceptance',
        prior_candidate={'version':'screw-appearance-v2','report_sha256':sha(previous/'report.json'),'rejected_reason':'one additional accepted known error in original regression'},
        scope='category retrieval only; fresh 40 frames excluded from all V1/V2 extraction, references and evaluation; older 301 and 473 images are development regression; one public dataset, physical identities unknown')
    incoming={'schema':1,'family_key':'screw','samples':dict(read(previous/'fresh_reservations.json')['samples'])}
    incoming['samples'].update({s['key']:{'experiment':'screw_appearance_v3','purpose':'fresh_appearance_evaluation',
        'source_frame':s['source_frame'],'pixel_hash':s['pixel_hash']} for s in fresh})
    shutil.copy2(OLD/'review_snapshot.json',output/'review_snapshot.json')
    write(output/'fresh_reservations.json',incoming);write(target,p);reserve(ROOT/'results',incoming)
    # All reused features have the exact same frozen encoder, preprocessing and image identities.
    cache=output/'features';cache.mkdir(exist_ok=True)
    for s in rows:
        source=previous/'features'/(s['key']+'.npz')
        if source.exists():shutil.copy2(source,cache/source.name)
    print({'version':p['version'],'fresh_frames':40,'fresh_images':len(fresh),'fixed_positive_gate':baseline['gate']},flush=True)


def gated_stats(rows,pred,categories,gate):
    masked=dict(pred);masked['similarity']=np.where(pred['boundary_gap']>=gate['exclusion_margin'],pred['similarity'],-2.)
    result=stats(rows,masked,categories,gate)
    result['excluded_by_boundary']=int(((pred['similarity']>=gate['similarity'])&(pred['boundary_gap']<gate['exclusion_margin'])).sum())
    return result


def promotion_report(measurements,rules):
    reasons=[]
    splits=['fresh_test','regression']+(['previous_fresh_regression'] if 'previous_fresh_regression' in measurements else [])
    for split in splits:
        before=measurements[split]['v1'];after=measurements[split]['v2']
        if after['known_correct_acceptance']+rules['max_known_correct_acceptance_drop']<before['known_correct_acceptance']:reasons.append(split+': known correct acceptance regressed')
        if after['accepted_known_wrong']>before['accepted_known_wrong']:reasons.append(split+': more accepted known errors')
        if after['other_rejection']<=before['other_rejection']:reasons.append(split+': no unknown rejection improvement')
    before=measurements['fresh_test']['v1'];after=measurements['fresh_test']['v2']
    if after['known_coverage']<rules['min_fresh_known_coverage']:reasons.append('fresh_test: known coverage below floor')
    if (after['accepted_precision_all'] or 0)<(before['accepted_precision_all'] or 0):reasons.append('fresh_test: accepted precision regressed')
    if measurements['control']['v2']['accepted_other']>measurements['control']['v1']['accepted_other']:reasons.append('control: more false accepts')
    return {'eligible':not reasons,'reasons':reasons}


def evaluate(output):
    if (output/'report.json').exists():raise ValueError('Completed boundary report is frozen')
    p=read(output/'protocol.json');baseline=AppearanceMatcher(OLD/'appearance.json')
    if baseline.meta['arrays_sha256']!=p['baseline_arrays_sha256']:raise ValueError('Baseline changed')
    def get(rows):
        x=[]
        for s in rows:
            with np.load(output/'features'/(s['key']+'.npz'),allow_pickle=False) as data:x.append(data['clip'])
        return np.asarray(x)
    refs=[s for s in p['samples'] if s['role']=='reference'];exclusions=[s for s in p['samples'] if s['role']=='exclusion']
    params={'clip':get(refs),'labels':np.array([s['category_id'] for s in refs]),'exclusion_clip':get(exclusions),
        'exclusion_labels':np.array([s['category_id'] for s in exclusions])}
    categories=[c['category_id'] for c in p['reference_categories']]
    def predict(rows):return boundary_ranked(get(rows),params['clip'],params['labels'],categories,params['exclusion_clip'])
    cal=[s for s in p['samples'] if s['role']=='validation'];pred=predict(cal);best=None
    cutoffs=[p['fixed_positive_gate']['similarity']] if p.get('fixed_positive_gate') else np.unique(np.r_[-1.,np.quantile(pred['similarity'],np.linspace(0,1,p['gate_grid']['score_quantiles'])),1.00001])
    cal_before=stats(cal,ranked(get(cal),baseline.params['clip'],baseline.params['labels'],categories),categories,baseline.meta['gate'])
    for cutoff in cutoffs:
        for gap in p['gate_grid']['exclusion_margin']:
            gate={'similarity':float(cutoff),'margin':0.,'exclusion_margin':float(gap)};m=gated_stats(cal,pred,categories,gate)
            if p.get('fixed_positive_gate') and m['known_correct_acceptance']+p['calibration_max_correct_acceptance_drop']<cal_before['known_correct_acceptance']:continue
            objective=.5*m['known_correct_acceptance']+.5*m['other_rejection'];key=(objective,m['known_coverage'],-abs(gap),-float(cutoff))
            if best is None or key>best[0]:best=(key,gate,m)
    if best is None:raise ValueError('No calibration gate met the conservative constraint')
    np.savez_compressed(output/'appearance.npz',**params)
    meta={**baseline.meta,'version':p['version'],'matcher_kind':'source-exclusion-v1','arrays_sha256':sha(output/'appearance.npz'),
        'protocol_sha256':sha(output/'protocol.json'),'gate':best[1],'exclusion_keys':[s['key'] for s in exclusions],
        'exclusion_categories':p['exclusion_categories'],'exclusion_reference_count':len(exclusions)}
    if p.get('fixed_positive_gate'):meta['fixed_positive_gate']=p['fixed_positive_gate']
    write(output/'appearance.json',meta)
    model=BoundaryAppearanceMatcher(output/'appearance.json');measurements={}
    for split in ['fresh_test','regression','control']+(['previous_fresh_regression'] if p.get('fixed_positive_gate') else []):
        rows=[s for s in p['samples'] if s['role']==split];pred=predict(rows)
        old_pred=ranked(get(rows),baseline.params['clip'],baseline.params['labels'],categories)
        measurements[split]={'v1':stats(rows,old_pred,categories,baseline.meta['gate']),'v2':gated_stats(rows,pred,categories,best[1])}
        for s,vector in zip(rows[:3],get(rows[:3])):assert model.predict(vector)['quality_assessment']=='not_available'
    gate=promotion_report(measurements,p['promotion'])
    report={'version':p['version'],'selected_method':'clip','reference_counts':p['reference_counts'],'reviewed_reference_count':len(refs),
        'exclusion_reference_count':len(exclusions),'exclusion_provenance':p['exclusion_provenance'],
        'primary_partition':'fresh_test','fresh_frame_count':40,'fresh_source':p['fresh_source'],'comparisons':measurements,'promotion':gate,
        'methods':{'clip':{'validation':best[2],'gate':best[1],'test':measurements['fresh_test']['v2'],'control':measurements['control']['v2']}},
        'arrays_sha256':meta['arrays_sha256'],'protocol_sha256':meta['protocol_sha256'],'quality_labels_used':False,'automatic_archival':False,
        'scope':p['scope'],'physical_ids_available':False}
    report['comparison_labels']={'v1':baseline.meta['version'],'v2':p['version']}
    if p.get('prior_candidate'):report['prior_candidate']=p['prior_candidate']
    write(output/'report.json',report)
    print({'gate':best[1],'promotion':gate,'comparisons':measurements},flush=True)


def audit(output):
    model=BoundaryAppearanceMatcher(output/'appearance.json');p=read(output/'protocol.json');decisions=[]
    for sample in p['samples']:
        if sample['role'] not in ('fresh_test','regression','previous_fresh_regression','control'):continue
        with np.load(output/'features'/(sample['key']+'.npz'),allow_pickle=False) as cache:value=model.predict(cache['clip'])
        decisions.append({'key':sample['key'],'role':sample['role'],'source_frame':sample['source_frame'],'path':sample['path'],
            'true_category':sample['category_id'],'matched_category':value['category_id'],'matched':value['matched'],
            'similarity':value['similarity'],'exclusion_similarity':value['exclusion_similarity'],'boundary_gap':value['boundary_gap'],
            'wrong_accept':bool(value['matched'] and value['category_id']!=sample['category_id'])})
    write(output/'decision_audit.json',{'version':model.meta['version'],'quality_labels_used':False,'posthoc_no_refit':True,'decisions':decisions})
    print({'wrong_accepts':dict(Counter(row['role'] for row in decisions if row['wrong_accept']))},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','prepare-conservative','extract','evaluate','audit']);parser.add_argument('--output',type=Path,default=ROOT/'results/experiments/screw_appearance_v2');args=parser.parse_args()
    {'prepare':prepare,'prepare-conservative':prepare_conservative,'extract':features,'evaluate':evaluate,'audit':audit}[args.action](args.output)


if __name__=='__main__':main()
