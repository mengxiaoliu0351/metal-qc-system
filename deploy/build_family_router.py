"""Package only frozen family routing references, reports and demonstration previews."""
import argparse
from pathlib import Path
import shutil

from experiments.prepare_m2_body import sha
from products.families import read,write

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();source=ROOT/'results/experiments/screw_family_router_v1'
    names=['router.json','router.npz','protocol.json','report.json','calibration_report.json','demo_samples.json','references.json']
    names+=[s['file'] for s in read(source/'demo_samples.json')['samples']]+[s['file'] for s in read(source/'references.json')['items']]
    for name in names:
        target=args.output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
    write(args.output/'router-manifest.json',{'directory':'screw_family_router_v1','files':{name:sha(args.output/name) for name in names}})
    print({'files':len(names),'bytes':sum((args.output/name).stat().st_size for name in names)},flush=True)


if __name__=='__main__':main()
