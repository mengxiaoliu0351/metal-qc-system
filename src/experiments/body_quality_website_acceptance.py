"""Validate candidate report, safety gate, demo imagery and actual quality inference."""
import argparse
from pathlib import Path
import time

import numpy as np
import requests

from perception.body_quality import BodyQualityModel
from products.families import read,write

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--directory',default='m2_body_quality_v1');args=parser.parse_args()
    folder=(ROOT/'results/experiments'/args.directory).resolve()
    if folder.parent!=(ROOT/'results/experiments').resolve():raise ValueError('Invalid local experiment directory')
    model=BodyQualityModel(folder/'quality.json');report=read(folder/'report.json');session=requests.Session();session.trust_env=False;base=args.base.rstrip('/')
    response=session.get(base+'/api/product-families/screw/body-quality',timeout=45);response.raise_for_status();d=response.json();assert d['ready'] and d['trial_ready']==report['eligible_for_research_trial']
    assert d['model']['arrays_sha256']==model.meta['arrays_sha256'];checks=[]
    for example in d['examples']:
        # Avoid repeatedly transferring multi-MB raw PNGs over the public link.
        # The server independently verifies the entire stored file hash first.
        with session.get(base+example['image_url'],headers={'Range':'bytes=0-1023'},stream=True,timeout=45) as image:
            image.raise_for_status();assert image.headers['content-type'].startswith('image/')
            assert image.raw.read(1024).startswith(b'\x89PNG\r\n\x1a\n')
        started=time.monotonic();response=session.post(base+example['image_url']+'/detect',timeout=120)
        if not report['eligible_for_research_trial']:
            assert response.status_code==409;checks.append({'id':example['id'],'blocked_by_gate':True});continue
        response.raise_for_status();actual=response.json()
        with np.load(folder/'features'/(example['id']+'.npz'),allow_pickle=False) as z:expected=model.predict(z['vector'],float(z['raw']))
        assert actual['scope_ok']==expected['scope_ok'] and actual['normal_defect_hint']==expected['normal_defect_hint']
        assert actual['automatic_archival'] is False and actual['requires_review'] is True
        if actual['scope_ok']:assert abs(actual['score']-expected['score'])<1e-4
        else:assert actual['score'] is None and actual['normal_defect_hint'] is None
        assert actual['heatmap'];checks.append({'id':example['id'],'scope_ok':actual['scope_ok'],'quality_hint':actual['normal_defect_hint'],'seconds':round(time.monotonic()-started,3)})
    assert len(checks)==6,'Expected all six prespecified demos'
    write(args.output,{'base':base,'version':report['version'],'passed':True,'trial_enabled':report['eligible_for_research_trial'],'checks':checks,'production_archival':False,
        'image_http_scope':'first 1024 PNG bytes via HTTP Range; server verifies full stored image hash','scope':'six prespecified demonstrations, not full website replay of primary test'})
    print({'passed':True,'examples':len(checks),'trial_enabled':report['eligible_for_research_trial']},flush=True)


if __name__=='__main__':main()
