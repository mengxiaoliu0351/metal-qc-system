"""Enable suitability dispatch only, preserving every production model and hold."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    temporary=p.with_suffix('.install.tmp');temporary.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8');os.chown(temporary,10001,10001);temporary.replace(p)

def passes(m,g):
    return m['known_accepted']/m['known']>=g['minimum_known_coverage'] and m['known_wrong_accepted']==0 and m['accepted_routing_accuracy']>=g['minimum_accepted_routing_accuracy'] and m['unknown_accepted']/m['unknown']<=g['maximum_unknown_acceptance'] and all(v['coverage']>=g['minimum_known_coverage'] for v in m['by_view'].values())

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args();stage=args.stage.resolve();root=Path('/opt/metalqc');results=root/'results'
    if not stage.is_relative_to(Path('/opt/metalqc-staging')):raise ValueError('Invalid staging directory')
    bundle=read(stage/'router-manifest.json');name=bundle['directory']
    if not re.fullmatch('screw_family_router_v[0-9]+',name):raise ValueError('Invalid routing version')
    for relative,expected in bundle['files'].items():
        path=(stage/relative).resolve()
        if not path.is_relative_to(stage) or sha(path)!=expected:raise ValueError('Routing bundle integrity failure')
    meta=read(stage/'router.json');p=read(stage/'protocol.json');r=read(stage/'report.json')
    if meta['family_key']!='screw' or meta['quality_decision_permission'] is not False or r['quality_decision_permission'] is not False:raise ValueError('Router cannot authorize quality release')
    if {v['key']:v['product_key'] for v in meta['views']}!={'mvtec_screw':'screw','m2_head':'product_a269405b7b9a','m2_body':None}:raise ValueError('Unverified quality destination')
    if sha(stage/'router.npz')!=meta['arrays_sha256'] or sha(stage/'protocol.json')!=meta['protocol_sha256'] or r['arrays_sha256']!=meta['arrays_sha256'] or r['protocol_sha256']!=meta['protocol_sha256']:raise ValueError('Routing protocol/model/report mismatch')
    if not r['operational_routing_ready'] or not passes(r['metrics'],p['release_gate']) or not passes(r['calibration']['metrics'],p['calibration_gate']):raise ValueError('Router did not pass per-view integration gates')
    registry=results/'perception/banks/products.json';before=sha(registry);target=results/'experiments'/name;target.mkdir(parents=True,exist_ok=True);os.chown(target,10001,10001)
    for relative in bundle['files']:
        destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and sha(destination)!=sha(stage/relative):raise ValueError('Do not overwrite frozen router evidence')
        if not destination.exists():shutil.copy2(stage/relative,destination)
        os.chown(destination,10001,10001);os.chown(destination.parent,10001,10001)
    path=results/'evaluation_reservations.json';reserved=read(path)
    for s in p['samples']:
        if s['role']=='test':reserved.setdefault('global',{}).setdefault(s['pixel_hash'],{'experiment':p['version'],'role':'routing_integration_test','scope':'routing regression, not quality accuracy'})
    write(path,reserved)
    write(results/'product_families/screw/operational_router.json',{'directory':'experiments/'+name,'version':meta['version'],'quality_decision_permission':False})
    if sha(registry)!=before:raise RuntimeError('Production models unexpectedly changed')
    print(json.dumps({'routing_enabled':True,'quality_models_changed':False,'body_candidate_enabled':False,'global_reservations':len(reserved['global'])}))

if __name__=='__main__':main()
