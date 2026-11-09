"""Safely extract and freeze publisher-labelled M2 BODY images before fitting."""
import argparse
from collections import Counter,defaultdict
import hashlib
from pathlib import Path,PureWindowsPath
import random

import numpy as np
from PIL import Image

from products.families import read,write

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/external/m2_v2'
EXPECTED='e660e1865310715451687db9657af26fd00da1f2fa69f91b10f5de80e17c4022'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def safe_target(destination,name):
    path=PureWindowsPath(name)
    if path.is_absolute() or path.drive or '..' in path.parts:raise ValueError('Unsafe archive path')
    target=(destination/Path(*path.parts)).resolve()
    if not target.is_relative_to(destination.resolve()):raise ValueError('Archive path escapes dataset directory')
    return target


def extract():
    import libarchive
    archive=DATA/'BODY.rar';destination=DATA/'body_source';marker=destination/'extracted.json'
    if sha(archive)!=EXPECTED:raise ValueError('Archive does not match publisher SHA256')
    if marker.exists() and read(marker)['sha256']==EXPECTED:return
    destination.mkdir(parents=True,exist_ok=True);count=total=0
    with libarchive.file_reader(str(archive)) as bundle:
        for entry in bundle:
            target=safe_target(destination,entry.pathname)
            if entry.issym or entry.islnk or not(entry.isdir or entry.isfile):raise ValueError('Links or special files forbidden in dataset archive')
            if entry.isdir:target.mkdir(parents=True,exist_ok=True);continue
            if target.suffix.lower() not in ('.bmp','.jpg','.jpeg','.png','.txt','.md','.json'):raise ValueError('Unexpected dataset archive file type')
            total+=entry.size
            if total>20*1024**3:raise ValueError('Unexpected unpacked dataset size')
            target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.extract.tmp');written=0
            with temporary.open('wb') as f:
                for chunk in entry.get_blocks():f.write(chunk);written+=len(chunk)
            if written!=entry.size:raise ValueError('Extracted file length differs')
            temporary.replace(target);count+=1
            if count%400==0:print({'extracted_files':count,'bytes':total},flush=True)
    write(marker,{'sha256':EXPECTED,'files':count,'unpacked_bytes':total});print({'extracted':count},flush=True)


def assign_roles(entries):
    groups=defaultdict(list)
    for item in entries:groups[item['pixel_hash']].append(item)
    for items in groups.values():
        if len({s['label'] for s in items})>1:
            for item in items:item['excluded']='conflicting_source_labels'
            continue
        items.sort(key=lambda s:(s['source_partition']!='Test',s['source_path']))
        for item in items[1:]:item['excluded']='duplicate_pixels'
    rng=random.Random(20261106)
    for label in ('normal','defect'):
        pool=[s for s in entries if not s['excluded'] and s['source_partition']=='Train' and s['label']==label];rng.shuffle(pool)
        required=480 if label=='normal' else 400
        if len(pool)<required:raise ValueError('Insufficient unique development images for declared budgets')
        bank=80 if label=='normal' else 0
        for index,item in enumerate(pool):
            item['role']='bank' if index<bank else 'calibration' if index<bank+100 else 'qualification' if index<bank+200 else 'train' if index<bank+400 else 'feedback'
    for item in entries:
        if not item['excluded'] and item['source_partition']=='Test':item['role']='test'
    return entries


def audit():
    source=DATA/'body_source/BODY'
    if not source.is_dir():raise ValueError('Expected BODY folder missing; inspect publisher archive structure')
    head=read(DATA/'head-manifest.json');head_hashes={s['pixel_hash'] for s in head['images']};entries=[];shapes=Counter();hashgroups=defaultdict(set)
    for path in sorted(source.rglob('*')):
        if path.suffix.lower() not in ('.bmp','.png','.jpg','.jpeg'):continue
        rel=path.relative_to(source)
        if len(rel.parts)!=3 or rel.parts[0] not in ('Train','Test') or rel.parts[1] not in ('OK','NG'):raise ValueError('Unknown BODY labels or source partition')
        with Image.open(path) as original:
            image=original.convert('RGB');fingerprint=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
            gray=np.asarray(image.convert('L').resize((9,8)));bits=(gray[:,1:]>gray[:,:-1]).ravel();dhash=sum(int(bit)<<i for i,bit in enumerate(bits));shapes[str(image.size)]+=1
        item={'source_path':path.relative_to(ROOT).as_posix(),'source_partition':rel.parts[0],
            'label':'normal' if rel.parts[1]=='OK' else 'defect','pixel_hash':fingerprint,'file_sha256':sha(path),
            'width':image.width,'height':image.height,'source_bytes':path.stat().st_size,'dhash':f'{dhash:016x}',
            'role':None,'excluded':'already_seen_head_pixels' if fingerprint in head_hashes else None}
        entries.append(item);hashgroups[item['dhash']].add(item['source_partition'])
        if len(entries)%400==0:print({'audited':len(entries)},flush=True)
    # DHash collisions are only a warning signal; they do not prove the same workpiece.
    entries=assign_roles(entries);counts=Counter((s['role'],s['label']) for s in entries if not s['excluded'])
    inspected={'Test/NG/BARI-1.BMP','Test/NG/BARI-10.BMP'}
    for item in entries:
        relative=Path(item['source_path']).relative_to(source.relative_to(ROOT)).as_posix()
        if relative in inspected and not item['excluded']:item['role']='layout_preview'
    counts=Counter((s['role'],s['label']) for s in entries if not s['excluded'])
    summary={'source_images':len(entries),'publisher_expected_images':3000,'source_count_matches_description':len(entries)==3000,
        'source_labels':dict(Counter(s['source_partition']+'/'+s['label'] for s in entries)),
        'retained':sum(not s['excluded'] for s in entries),'excluded':dict(Counter(s['excluded'] for s in entries if s['excluded'])),
        'cross_head_pixel_matches':sum(s['pixel_hash'] in head_hashes for s in entries),'dimensions':dict(shapes),
        'cross_source_partition_dhash_collisions':sum(len(partitions)>1 for partitions in hashgroups.values()),
        'roles':{r:{label:counts[r,label] for label in ('normal','defect')} for r in ('bank','train','calibration','qualification','test','layout_preview','feedback')}}
    value={'dataset':'M2 screw defect detection dataset V2, BODY','doi':'10.17632/2y6mcsyrpy.2','source':'https://data.mendeley.com/datasets/2y6mcsyrpy/2',
        'license':'CC BY 4.0','archive_sha256':EXPECTED,'physical_identity':'not provided; pixel de-duplication does not establish independent workpieces',
        'label_basis':'publisher OK/NG folder labels; no quality labels generated by appearance matching','layout_preview_paths':sorted(inspected),
        'layout_preview_note':'two test files rendered for layout inspection before protocol freeze; excluded from primary sealed test','seed':20261106,'summary':summary,'images':entries}
    output=DATA/'body-manifest.json'
    if output.exists() and read(output)!=value:raise ValueError('Existing BODY manifest differs; do not overwrite frozen source audit')
    write(output,value);print(summary,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['extract','audit']);args=parser.parse_args()
    {'extract':extract,'audit':audit}[args.action]()


if __name__=='__main__':main()
