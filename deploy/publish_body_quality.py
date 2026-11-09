"""Install BODY research report/trial without changing production product recipes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path,value):
    temporary=path.with_suffix('.install.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');os.chown(temporary,10001,10001);temporary.replace(path)


def validate_trial_gate(meta,report,protocol):
    """Independently veto publication if any required cohort failed."""
    gate=protocol['sealed_trial_gate']
    if gate['minimum_recall']<.9 or gate['maximum_false_positive_rate']>.1 or gate['minimum_scope_coverage']<.95 or gate['minimum_head_rejection']<.95:
        raise ValueError('Do not relax BODY research publication floors')
    def passed(metrics,coverage,limits):
        return metrics['recall']>=limits['minimum_recall'] and metrics['false_positive_rate']<=limits['maximum_false_positive_rate'] and coverage>=limits['minimum_scope_coverage']
    allowed=bool(meta['qualified_for_research_trial']) and passed(report['methods'][meta['selected_mode']],report['scope_coverage'],gate)
    controls=report['head_controls']
    allowed=allowed and controls['n']>0 and 1-controls['incorrectly_accepted_as_body']/controls['n']>=gate['minimum_head_rejection']
    if 'historical_regression_gate' in protocol:
        limits=protocol['historical_regression_gate'];history=report['historical_regression'];q=report['qualification']
        if limits['minimum_recall']<.9 or limits['maximum_false_positive_rate']>.1 or limits['minimum_scope_coverage']<.95:raise ValueError('Do not relax regression floors')
        allowed=allowed and passed(history['current'],history['scope_coverage'],limits) and passed(q['metrics'],q['scope_coverage'],protocol['qualification_gate'])
    if bool(report['eligible_for_research_trial'])!=bool(allowed) or bool(report['trial_gate_reasons'])==bool(allowed):
        raise ValueError('Declared trial permission does not match cohort gates')
    return bool(allowed)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args();stage=args.stage.resolve();root=Path('/opt/metalqc');results=root/'results'
    if not stage.is_relative_to(Path('/opt/metalqc-staging')):raise ValueError('Unexpected staging directory')
    bundle=read(stage/'body-quality-manifest.json');name=bundle['directory']
    if not re.fullmatch('m2_body_quality_v[0-9]+',name):raise ValueError('Invalid BODY experiment version')
    for relative,expected in bundle['files'].items():
        path=(stage/relative).resolve()
        if not path.is_relative_to(stage) or sha(path)!=expected:raise ValueError('Staged file integrity failure')
    meta=read(stage/'quality.json');report=read(stage/'report.json');protocol=read(stage/'protocol.json');source=read(stage/'source-audit.json')
    if meta['family_key']!='screw' or meta['view']!='m2_body' or meta['quality_label_basis']!='publisher_OK_NG' or meta['automatic_archival'] is not False or report['automatic_archival'] is not False:raise ValueError('BODY candidate cannot grant production permission')
    if sha(stage/'quality.npz')!=meta['arrays_sha256'] or sha(stage/'protocol.json')!=meta['protocol_sha256'] or report['arrays_sha256']!=meta['arrays_sha256'] or report['protocol_sha256']!=meta['protocol_sha256']:raise ValueError('Unpaired quality model/report')
    validate_trial_gate(meta,report,protocol)
    if protocol['body_manifest_sha256']!=sha(stage/'source-audit.json') or read(stage/'archive-receipt.json')['sha256']!=source['archive_sha256']:raise ValueError('Source audit mismatch')
    official={s['pixel_hash'] for s in protocol['samples'] if s['role'] in ('test','layout_preview')}
    if set(read(stage/'test_reservations.json')['pixel_hashes'])!=official:raise ValueError('Test reservations do not match protocol')
    for sample in read(stage/'demo_samples.json')['samples']:
        if sample['id'] not in official and sample['source_view']!='head':raise ValueError('Demo was not sampled from declared test source')
        if sha(stage/sample['file'])!=sample['file_sha256']:raise ValueError('Demo image changed')
    registry=results/'perception/banks/products.json';before=sha(registry);target=results/'experiments'/name;target.mkdir(parents=True,exist_ok=True);os.chown(target,10001,10001)
    for relative in bundle['files']:
        destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and sha(destination)!=sha(stage/relative):raise ValueError('Do not overwrite frozen quality artifacts')
        if not destination.exists():shutil.copy2(stage/relative,destination)
        os.chown(destination,10001,10001);os.chown(destination.parent,10001,10001)
    reservation_path=results/'evaluation_reservations.json';reserved=read(reservation_path) if reservation_path.exists() else {'version':1,'products':{}}
    for sample in protocol['samples']:
        if sample['pixel_hash'] in official:reserved.setdefault('global',{}).setdefault(sample['pixel_hash'],{'experiment':protocol['version'],'role':sample['role'],'dataset':'M2 BODY V2'})
    atomic(reservation_path,reserved)
    pointer=results/'product_families/screw/body_quality_candidate.json';atomic(pointer,{'directory':'experiments/'+name,'version':meta['version'],'report_only':not report['eligible_for_research_trial'],'automatic_archival':False})
    if sha(registry)!=before:raise RuntimeError('Production registry unexpectedly changed')
    print(json.dumps({'installed':name,'research_trial':report['eligible_for_research_trial'],'test_reserved':len(official),'production_unchanged':True,'automatic_archival':False}))


if __name__=='__main__':main()
