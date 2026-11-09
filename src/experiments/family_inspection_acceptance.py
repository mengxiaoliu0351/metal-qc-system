"""Five real HTTP integrations: automatic dispatch, pending exceptions, history, PDF.

Fixed examples are explicitly simulation records, without human labels or training approval.
"""
import argparse
from pathlib import Path
import time
import requests

from products.families import write

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    session=requests.Session();session.trust_env=False;base=args.base.rstrip('/');path='/api/product-families/screw/inspection'
    response=session.get(base+path,timeout=60);response.raise_for_status();config=response.json();assert config['ready'];checks=[]
    for sample in config['examples']:
        started=time.monotonic();response=session.post(base+sample['image_url']+'/detect',timeout=180);response.raise_for_status();r=response.json();assert r['session_id'] and r['disposition']
        routing=r.get('family_routing') or {}
        if routing.get('matched'):assert routing['view_key']==sample['expected_view']
        if not r['quality_model_bound']:
            assert r['score'] is None and r['perception_prediction'] is None and not r['vlm_called'] and r['iteration_eligible'] is False
            assert r['disposition']['requires_review']
        assert r['disposition']['requires_review'],'Fixed examples must be simulation reviews'
        pdf=session.get(base+'/api/report/'+r['session_id'],timeout=60);pdf.raise_for_status();assert pdf.content.startswith(b'%PDF')
        checks.append({'sample':sample['label'],'expected_view':sample['expected_view'],'routed_view':routing.get('view_key'),
            'session_id':r['session_id'],'family_action':r['family_action'],'quality_model_bound':r['quality_model_bound'],
            'vlm_called':r['vlm_called'],'requires_review':r['disposition']['requires_review'],'pdf_valid':True,'seconds':round(time.monotonic()-started,3)})
        print({'sample':sample['label'],'action':r['family_action'],'pdf_valid':True},flush=True)
    assert len(checks)==5 and any(c['family_action']=='product_workflow' for c in checks)
    assert all(not c['quality_model_bound'] for c in checks if c['expected_view'] in (None,'m2_body'))
    response=session.get(base+path+'/metrics',timeout=45);response.raise_for_status();summary=response.json();assert summary['ok']
    write(args.output,{'base':base,'passed':True,'checks':checks,'scope':'five fixed simulations, not full quality test replay; no human labels submitted','metrics_available':True})
    print({'passed':True,'examples':len(checks)},flush=True)

if __name__=='__main__':main()
