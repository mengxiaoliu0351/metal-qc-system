"""Install a frozen appearance-only reference set, never a quality model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args();stage=args.stage.resolve()
    root=Path('/opt/metalqc');results=root/'results';family=results/'product_families/screw'
    if not stage.is_relative_to(Path('/opt/metalqc-staging')):raise ValueError('Unexpected staging directory')
    manifest=read(stage/'appearance-manifest.json');name=manifest['directory']
    if not re.fullmatch(r'screw_appearance_v[0-9]+',name):raise ValueError('Unexpected appearance version')
    files=['appearance.json','appearance.npz','protocol.json','report.json','review_snapshot.json']
    for file in files:
        if sha(stage/file)!=manifest['files'][file]:raise ValueError('Staged checksum mismatch: '+file)
    meta=read(stage/'appearance.json');protocol=read(stage/'protocol.json');report=read(stage/'report.json')
    if meta.get('matcher_kind')=='source-exclusion-v1':
        if not report.get('promotion',{}).get('eligible'):raise ValueError('Candidate did not pass predeclared non-regression gates')
        if sha(stage/'fresh_reservations.json')!=manifest['files'].get('fresh_reservations.json'):raise ValueError('Fresh holdout reservation mismatch')
        files.append('fresh_reservations.json')
    if any(x['quality_labels_used'] is not False or x['automatic_archival'] is not False for x in (meta,protocol,report)):raise ValueError('Appearance package cannot grant quality permission')
    if meta['arrays_sha256']!=sha(stage/'appearance.npz') or meta['protocol_sha256']!=sha(stage/'protocol.json'):raise ValueError('Unpaired artifacts')
    if report['arrays_sha256']!=meta['arrays_sha256'] or report['protocol_sha256']!=meta['protocol_sha256']:raise ValueError('Unpaired report')
    if protocol['manifest_sha256']!=sha(family/'appearance.json') or protocol['review_snapshot_sha256']!=sha(stage/'review_snapshot.json'):raise ValueError('Source or review snapshot changed')
    checks=read(family/'appearance_checks.json');samples={s['key']:s for s in read(family/'appearance.json')['samples']}
    reservation_path=family/'appearance_reservations.json'
    reserved=read(reservation_path) if reservation_path.exists() else {'schema':1,'family_key':'screw','samples':{}}
    if meta.get('matcher_kind')=='source-exclusion-v1':
        incoming=read(stage/'fresh_reservations.json')
        expected={s['key'] for s in protocol['samples'] if s['role'] in ('fresh_test','previous_fresh_regression')}
        if set(incoming['samples'])!=expected:raise ValueError('Fresh reservation set does not match protocol')
        for key,value in incoming['samples'].items():
            if key in reserved['samples'] and reserved['samples'][key]!=value:raise ValueError('Frozen reservation conflict')
        reserved['samples'].update(incoming['samples'])
    frozen_frames={s['source_frame'] for s in reserved['samples'].values()}
    for key in meta['reference_keys']:
        if key not in samples or samples[key]['split']!='train' or checks.get(key,{}).get('appearance_status')!='accepted':raise ValueError('Reference is not currently approved as appearance material')
        if key in reserved['samples'] or samples[key]['source_frame'] in frozen_frames:raise ValueError('Frozen evaluation image cannot enter appearance references')
    exclusions={s['key']:s for s in protocol['samples'] if s['role']=='exclusion'}
    if set(meta.get('exclusion_keys',[]))!=set(exclusions):raise ValueError('Exclusion reference identities differ')
    supported={c['category_id'] for c in meta['categories']}
    for key,sample in exclusions.items():
        if samples[key]['split']!='train' or samples[key]['category_id'] in supported or samples[key]['category_id']!=sample['category_id']:raise ValueError('Invalid source category exclusion')
        if key in reserved['samples'] or sample['source_frame'] in frozen_frames:raise ValueError('Fresh evaluation leaked into exclusions')
    registry=results/'perception/banks/products.json';production_before=sha(registry)
    destination=results/'experiments'/name;destination.mkdir(parents=True,exist_ok=True);os.chown(destination,10001,10001)
    for file in files:
        target=destination/file
        if target.exists() and sha(target)!=sha(stage/file):raise ValueError('Do not overwrite frozen appearance artifacts')
        if not target.exists():shutil.copy2(stage/file,target)
        os.chown(target,10001,10001)
    if meta.get('matcher_kind')=='source-exclusion-v1':
        temporary=reservation_path.with_suffix('.tmp');temporary.write_text(json.dumps(reserved,ensure_ascii=False,indent=2),encoding='utf8');os.chown(temporary,10001,10001);temporary.replace(reservation_path)
    active=family/'appearance_active.json';previous=read(active) if active.exists() else None
    receipt={'directory':'experiments/'+name,'version':meta['version'],'automatic_archival':False,'quality_assessment':'not_available','previous':previous}
    temporary=active.with_suffix('.tmp');temporary.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8');os.chown(temporary,10001,10001);temporary.replace(active)
    if sha(registry)!=production_before:raise RuntimeError('Production registry unexpectedly changed')
    print(json.dumps({'installed':name,'references':len(meta['reference_keys']),'production_unchanged':True,'quality_permission':False}))


if __name__=='__main__':main()
