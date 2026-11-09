"""Compare actual family website inference with frozen local feature predictions."""
import argparse
import io
from pathlib import Path
import sys

import requests
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from screw_family import FamilyModel,Features,CONDITIONS,transform
from review.disposition import image_quality
from review.workflow import read,write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--endpoint',default='http://127.0.0.1:8000');parser.add_argument('--experiment',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    session=requests.Session();session.trust_env=False;base=args.endpoint.rstrip('/')
    protocol=read(args.experiment/'protocol.json');model=FamilyModel(args.experiment/'family.json');cache=Features(args.experiment,protocol)
    samples=[]
    for group in (0,1):
        for label in (0,1):
            candidates=[s for s in protocol['samples'] if s['group']==group and s['label']==label and s['role']=='new_test']
            if not candidates:candidates=[s for s in protocol['samples'] if s['group']==group and s['label']==label and s['role']=='test']
            samples.append(candidates[0])
    before=session.get(base+'/api/products',timeout=30);before.raise_for_status();products=before.json()['products'];checks=[]
    for condition in CONDITIONS:
        for sample in samples:
            with Image.open(ROOT/sample['path']) as original:image=transform(original,condition)
            content=io.BytesIO();image.save(content,format='PNG')
            response=session.post(base+'/api/product-family/detect',files={'file':('unlabelled.png',content.getvalue(),'image/png')},timeout=90)
            quality=image_quality(image)
            if not quality['ok']:
                assert response.status_code==400,response.text
                checks.append({'condition':condition,'group':sample['group'],'quality_rejected':True});continue
            response.raise_for_status();actual=response.json();_,vector=cache.get(sample,condition);expected=model.predict(vector)
            assert actual['routing']['group']==expected['routing']['group'],actual
            assert actual['routing']['in_scope']==expected['routing']['in_scope'],actual
            assert actual['predictions']==expected['predictions'],actual
            difference=max(abs(actual['scores'][a]-expected['scores'][a]) for a in expected['scores'])
            assert difference<.001,difference
            assert actual['requires_review'] and not actual['automatic_archival'] and actual['heatmap']
            assert actual['version']==model.meta['version'],actual
            checks.append({'condition':condition,'group':sample['group'],'label':sample['label'],'max_score_difference':difference,
                           'routed_group':actual['routing']['group'],'in_scope':actual['routing']['in_scope'],'automatic_archival':False})
            print(checks[-1],flush=True)
    for color in ('black','white'):
        data=io.BytesIO();Image.new('RGB',(128,128),color).save(data,format='PNG')
        response=session.post(base+'/api/product-family/detect',files={'file':('blank.png',data.getvalue(),'image/png')},timeout=30)
        assert response.status_code==400,response.text
        checks.append({'blank':color,'http_status':400})
    after=session.get(base+'/api/products',timeout=30);after.raise_for_status();assert products==after.json()['products']
    benchmark=session.get(base+'/api/product-family/benchmark',timeout=30);benchmark.raise_for_status();assert benchmark.json()['ready']
    result={'endpoint':base,'version':model.meta['version'],'checks':checks,'passed':True,'original_products_unchanged':True,
            'scope':'actual API inference/quality checks; no automatic archival; not a full browser visual inspection'}
    write(args.output,result);print('Family website inference acceptance passed',flush=True)


if __name__=='__main__':main()
