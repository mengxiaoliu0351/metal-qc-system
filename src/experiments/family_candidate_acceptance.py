"""Read-only local/cloud acceptance for the frozen appearance-candidate phase."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

from PIL import Image
import requests

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads((ROOT/'results/product_families/screw/appearance.json').read_text(encoding='utf8'))
    session=requests.Session();session.trust_env=False;checks=[]
    def get(path):
        r=session.get(args.base.rstrip('/')+path,timeout=45);r.raise_for_status();checks.append({'path':path,'status':r.status_code});return r
    families=get('/api/product-families').json()['families'];family=next(f for f in families if f['key']=='screw')
    assert family['appearance_summary']==manifest['summary']
    assert family['reviewed_normal']==0 and family['eligible_normal_references']==0,'Do not replay acceptance over real operator reviews'
    for kind in manifest['types']:
        result=get('/api/product-families/screw/appearance?category='+str(kind['category_id'])+'&limit=1').json()
        assert result['count']==kind['count'];sample=result['samples'][0]
        assert sample['quality_label']=='unverified' and not sample['eligible_normal_reference']
        raw=get(sample['image_url']).content
        assert hashlib.sha256(raw).hexdigest()==sample['file_sha256']
        with Image.open(io.BytesIO(raw)) as image:assert image.format=='PNG' and max(image.size)<=320
    assert get('/api/product-families/screw/appearance?status=normal').json()['count']==0
    assert get('/api/product-families/screw/appearance?offset=3049').json()['samples']==[]
    benchmark=get('/api/product-family/benchmark?family=screw').json()
    assert benchmark['ready'] and benchmark['automatic_archival'] is False
    report={'base':args.base,'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'checks':checks,'summary':manifest['summary'],
        'appearance_sha256':hashlib.sha256((ROOT/'results/product_families/screw/appearance.json').read_bytes()).hexdigest(),
        'active_family_version':benchmark['model']['version'],'candidate_quality_labels_inferred':False,'read_only':True,'passed':True}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'passed':True,'checks':len(checks),'summary':manifest['summary']},ensure_ascii=False))


if __name__=='__main__':main()
