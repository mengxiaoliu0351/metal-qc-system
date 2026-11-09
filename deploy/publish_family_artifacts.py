"""Install checksummed advisory-family artifacts without touching product models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args()
    root=Path('/opt/metalqc');stage=args.stage.resolve()
    if not stage.is_relative_to(Path('/opt/metalqc-staging')) or not (root/'results').is_dir():raise ValueError('Unexpected deployment location')
    manifest=json.loads((stage/'family-manifest.json').read_text());name=manifest['directory']
    if not re.fullmatch(r'screw_family_v[0-9]+',name):raise ValueError('Invalid family version directory')
    names=('family.json','family.npz','protocol.json','report.json','fresh_report.json','evaluation_reservations.json')
    for file in names:
        if sha(stage/file)!=manifest['files'][file]['sha256']:raise ValueError('Staged family checksum mismatch: '+file)
    meta=json.loads((stage/'family.json').read_text());report=json.loads((stage/'report.json').read_text());fresh=json.loads((stage/'fresh_report.json').read_text())
    if meta['automatic_archival'] or report['automatic_archival'] or fresh['automatic_archival']:raise ValueError('Only advisory family models may be published by this script')
    if meta['arrays_sha256']!=sha(stage/'family.npz') or meta['protocol_sha256']!=sha(stage/'protocol.json'):
        raise ValueError('Family metadata is not paired with arrays and protocol')
    if any(r['arrays_sha256']!=meta['arrays_sha256'] or r['protocol_sha256']!=meta['protocol_sha256'] for r in (report,fresh)):
        raise ValueError('Family report is not paired with the model')
    destination=root/'results/experiments'/name
    destination.mkdir(parents=True,exist_ok=True);os.chown(destination,10001,10001)
    for file in names:
        if file=='evaluation_reservations.json':continue
        target=destination/file
        if target.exists() and sha(target)!=sha(stage/file):raise ValueError('Do not overwrite an existing family artifact version')
        if not target.exists():shutil.copy2(stage/file,target)
        os.chown(target,10001,10001)
    reservations_path=root/'results/evaluation_reservations.json'
    incoming=json.loads((stage/'evaluation_reservations.json').read_text())
    reservations=json.loads(reservations_path.read_text()) if reservations_path.exists() else {'version':1,'products':{}}
    for product,images in incoming['products'].items():
        reservations['products'].setdefault(product,{}).update(images)
    temporary=reservations_path.with_suffix('.family.tmp');temporary.write_text(json.dumps(reservations,ensure_ascii=False,indent=2),encoding='utf8');os.chown(temporary,10001,10001);temporary.replace(reservations_path)
    active=root/'results/perception/family_active.json'
    previous=json.loads(active.read_text()) if active.exists() else None
    if previous and previous.get('directory')==name and previous.get('model_sha256')==meta['arrays_sha256']:
        print(json.dumps({'installed':name,'already_active':True,'advisory_only':True}));return
    receipt={'directory':name,'version':meta['version'],'automatic_archival':False,'model_sha256':meta['arrays_sha256'],
             'fresh_gate_passed':fresh['fresh_gate_passed'],'previous':previous}
    temporary=active.with_suffix('.family.tmp');temporary.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8');os.chown(temporary,10001,10001);temporary.replace(active)
    print(json.dumps({'installed':name,'advisory_only':True,'fresh_gate_passed':fresh['fresh_gate_passed'],'original_products_untouched':True}))


if __name__=='__main__':main()
