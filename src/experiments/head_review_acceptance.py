"""Three real HTTP HEAD routes, scoring parity, mandatory review, history and PDF.

Use only against isolated storage: these uploads deliberately exercise the actual
policy without the simulation override. No human verdict or training approval.
"""
import argparse
from pathlib import Path

import requests

from experiments.m2_head_expand import KEY,OUTPUT,ROOT
from products.families import read,write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.base.rstrip('/')
    if base!='http://127.0.0.1:8011':raise ValueError('Actual-policy probe is restricted to the isolated local server')
    session=requests.Session();session.trust_env=False
    meta=read(OUTPUT/'candidate.json');low,high=meta['thresholds'];p=read(OUTPUT/'protocol.json')
    sources={s['pixel_hash']:s for s in p['samples'] if s['role']=='test'}
    previous=[read(path) for path in (ROOT/'results/experiments/screw_entry_regression_v1/cases').glob('*.json')]
    matched={r['sample']['pixel_hash'] for r in previous if r['sample']['cohort']=='m2_head' and r['valid_quality_prediction']}
    predictions=[r for r in read(OUTPUT/'predictions.json') if r['role']=='test' and r['pixel_hash'] in matched]
    route=lambda score:'low_risk' if score<low else 'high_risk' if score>high else 'vlm_review'
    samples=[min((r for r in predictions if route(r['candidate_score'])==kind),key=lambda r:r['pixel_hash']) for kind in ('low_risk','vlm_review','high_risk')]
    checks=[]
    for sample in samples:
        source=sources[sample['pixel_hash']]
        with (ROOT/source['path']).open('rb') as f:
            response=session.post(base+'/api/product-families/screw/inspection/detect',files={'file':(sample['pixel_hash'][:16]+'.png',f,'image/png')},timeout=180)
        response.raise_for_status();r=response.json()
        assert r['family_routing']['view_key']=='m2_head' and r['quality_model_bound']
        # Cached batched and live single-image CLIP run in float32; allow a
        # bounded encoder roundoff, while requiring identical route decisions.
        assert abs(r['score']-sample['candidate_score'])<1e-4,(r['score'],sample['candidate_score'])
        assert r['threshold']==low and r['t_high']==high
        assert r['route']==route(sample['candidate_score'])
        action=r['disposition'];assert action['requires_review'] and action.get('auto_result') is None and action['reason_code']=='quality_hold'
        if r['vlm_called']:assert r['cognition']['ok'],'Real VLM roundtrip failed'
        loc=session.get(base+'/api/review/record',params={'session_id':r['session_id']},timeout=30);loc.raise_for_status()
        history=session.get(base+'/api/history/detail',params={k:loc.json()[k] for k in ('day','key')},timeout=30);history.raise_for_status()
        data=history.json();record=data.get('record',data)
        assert record['perception_version']=='head_expanded_20261108',data.keys()
        pdf=session.get(base+'/api/report/'+r['session_id'],timeout=60);pdf.raise_for_status();assert pdf.content.startswith(b'%PDF')
        checks.append({'route':r['route'],'score':r['score'],'expected_score':sample['candidate_score'],'session_id':r['session_id'],
                       'requires_review':True,'hold_reason':action['reason_code'],'vlm_called':r['vlm_called'],'pdf_valid':True,'history_version_valid':True})
        print({'route':r['route'],'score_parity':True,'quality_hold':True,'pdf_valid':True},flush=True)
    write(args.output,{'passed':True,'base':base,'checks':checks,'scope':'isolated HTTP routing/gray analysis/review/history/PDF acceptance, not new quality accuracy','human_labels_submitted':False})


if __name__=='__main__':main()
