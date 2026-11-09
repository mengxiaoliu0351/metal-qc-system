"""Package only frozen appearance candidates; never operator reviews or models."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    scope=ROOT/'results/product_families/screw';appearance=scope/'appearance.json'
    data=json.loads(appearance.read_text(encoding='utf8'));files={appearance}
    if len({s['key'] for s in data['samples']})!=len(data['samples']):raise ValueError('Duplicate sample IDs')
    for sample in data['samples']:
        path=(ROOT/'results'/sample['path']).resolve()
        if not path.is_relative_to(scope.resolve()) or sha(path)!=sample['file_sha256']:raise ValueError('Candidate image integrity failure')
        files.add(path)
    files.update(scope/'mvtec_screws_v1'/name for name in ('audit.json','alignment-contact.jpg'))
    if sum(p.stat().st_size for p in files)>250*1024**2:raise ValueError('Candidate set exceeds deployment allowance')
    args.output.mkdir(parents=True,exist_ok=True);archive=args.output/'candidates.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as bundle:
        for path in sorted(files):bundle.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    manifest={'archive_sha256':sha(archive),'files':{p.relative_to(ROOT).as_posix():sha(p) for p in files},'summary':data['summary']}
    (args.output/'candidates-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'archive_bytes':archive.stat().st_size,'files':len(files),'summary':data['summary']},ensure_ascii=False))


if __name__=='__main__':main()
