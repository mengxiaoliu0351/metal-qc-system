"""Bundle frozen BODY quality research and a few exact-pixel demo images."""
import argparse
import hashlib
from pathlib import Path
import shutil
import re

from products.families import read,write

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--directory',default='m2_body_quality_v1');args=parser.parse_args()
    if not re.fullmatch('m2_body_quality_v[0-9]+',args.directory):raise ValueError('Invalid BODY experiment directory')
    source=ROOT/'results/experiments'/args.directory
    names=['quality.json','quality.npz','protocol.json','report.json','development_report.json','test_reservations.json','demo_samples.json']
    names.extend(name for name in ('qualification_report.json','candidate_models.npz') if (source/name).exists())
    names.extend(s['file'] for s in read(source/'demo_samples.json')['samples']);args.output.mkdir(parents=True,exist_ok=True)
    for name in names:
        path=(source/name).resolve()
        if not path.is_relative_to(source.resolve()):raise ValueError('Unexpected bundle path')
        target=args.output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    shutil.copy2(ROOT/'data/external/m2_v2/body-manifest.json',args.output/'source-audit.json');names.append('source-audit.json')
    shutil.copy2(ROOT/'data/external/m2_v2/body-download-receipt.json',args.output/'archive-receipt.json');names.append('archive-receipt.json')
    write(args.output/'body-quality-manifest.json',{'directory':args.directory,'files':{name:sha(args.output/name) for name in names}})
    print({'files':len(names),'bytes':sum((args.output/n).stat().st_size for n in names)},flush=True)


if __name__=='__main__':main()
