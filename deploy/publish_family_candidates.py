"""Install a verified candidate set; preserve human reviews and production registry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args()
    stage=args.stage.resolve();root=Path('/opt/metalqc');scope=root/'results/product_families/screw'
    if not stage.is_relative_to(Path('/opt/metalqc-staging')):raise ValueError('Unexpected staging location')
    manifest=json.loads((stage/'candidates-manifest.json').read_text())
    archive=stage/'candidates.tar.gz'
    if sha(archive)!=manifest['archive_sha256']:raise ValueError('Archive checksum failed')
    production=root/'results/perception/banks/products.json';before=sha(production)
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers()
        if sum(m.size for m in members)>250*1024**2:raise ValueError('Candidate set exceeds deployment allowance')
        for item in members:
            target=(root/item.name).resolve()
            if not item.isfile() or not target.is_relative_to(scope) or item.name not in manifest['files']:raise ValueError('Unexpected archive entry')
            if target.exists() and sha(target)!=manifest['files'][item.name]:raise ValueError('Existing candidate differs; use explicit data version migration')
            data=bundle.extractfile(item).read()
            if hashlib.sha256(data).hexdigest()!=manifest['files'][item.name]:raise ValueError('Candidate checksum mismatch')
        if len(members)!=len(manifest['files']):raise ValueError('Incomplete candidate archive')
        # Publish the manifest last, after every image is readable.
        for item in sorted(members,key=lambda m:m.name.endswith('/appearance.json')):
            target=root/item.name
            if target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            for parent in target.parents:
                if not parent.is_relative_to(root/'results/product_families'):break
                os.chown(parent,10001,10001)
            temporary=target.with_name(target.name+'.candidate.tmp')
            temporary.write_bytes(bundle.extractfile(item).read());os.chown(temporary,10001,10001);temporary.replace(target)
    for name in manifest['files']:
        target=root/name;os.chown(target,10001,10001)
        for parent in target.parents:
            if not parent.is_relative_to(root/'results/product_families'):break
            os.chown(parent,10001,10001)
    if sha(production)!=before:raise RuntimeError('Production registry unexpectedly changed')
    print(json.dumps({'installed_files':len(members),'production_registry_unchanged':True,'quality_labels_assumed':False,'model_changed':False}))


if __name__=='__main__':main()
