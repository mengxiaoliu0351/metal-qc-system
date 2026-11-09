"""Prepare category-only aligned candidates; never infer quality from category."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path

from PIL import Image,ImageDraw

from products.families import write

ROOT=Path(__file__).resolve().parents[2]
ALIASES={1:'长杆螺丝A',2:'细长螺丝',3:'长杆螺丝B',4:'短杆螺丝',5:'亮面螺丝',6:'黑色螺丝',
         7:'螺母外观7',8:'螺栓',9:'大螺母',10:'螺母外观10',11:'螺母外观11',12:'机螺丝',13:'短机螺丝'}
SCREW_IDS={1,2,3,4,5,6,8,12,13}


def pixel_hash(image):
    image=image.convert('RGB')
    return hashlib.sha256(f'{image.width}x{image.height}:RGB:'.encode()+image.tobytes()).hexdigest()


def aligned_crop(image,bbox,margin=1.16,maximum=320):
    """MVTec: row, column, length1, length2, phi; rotate long length1 to vertical.

    Uses an inverse affine mapping directly from the source, with one resample.
    This requires a supplied annotation box; it is NOT a localization detector.
    """
    row,column,length1,length2,phi=map(float,bbox)
    if not all(math.isfinite(v) for v in bbox) or min(length1,length2)<=0:raise ValueError('Invalid oriented box')
    width,height=length2*margin,length1*margin
    scale=min(1.,maximum/max(width,height));out_w=max(2,round(width*scale));out_h=max(2,round(height*scale))
    sx=width/out_w;sy=height/out_h;c=math.sin(phi);s=math.cos(phi)
    # theta=pi/2-phi: horizontal axis=(c,s), vertical axis=(-s,c).
    a,b=c*sx,-s*sy;d,e=s*sx,c*sy
    affine=(a,b,column-a*out_w/2-b*out_h/2,d,e,row-d*out_w/2-e*out_h/2)
    return image.convert('RGB').transform((out_w,out_h),Image.Transform.AFFINE,affine,
        resample=Image.Resampling.BICUBIC,fillcolor=(240,240,240))


def group_splits(samples):
    """Union source frames linked by exact duplicate crop pixels before splitting."""
    parent={s['source_frame']:s['source_frame'] for s in samples}
    def find(key):
        while parent[key]!=key:parent[key]=parent[parent[key]];key=parent[key]
        return key
    hashes={}
    for sample in samples:
        key=sample['pixel_hash'];frame=sample['source_frame']
        if key in hashes:
            a,b=find(frame),find(hashes[key]);parent[max(a,b)]=min(a,b)
        else:hashes[key]=frame
    groups=defaultdict(list)
    for frame in parent:groups[find(frame)].append(frame)
    ordered=sorted(groups,key=lambda g:hashlib.sha256(('metalqc-screws-frame-v1:'+g).encode()).hexdigest())
    train=round(len(ordered)*.7);validation=round(len(ordered)*.15)
    assigned={g:('train' if i<train else 'validation' if i<train+validation else 'test') for i,g in enumerate(ordered)}
    for sample in samples:sample['split']=assigned[find(sample['source_frame'])];sample['split_group']=find(sample['source_frame'])
    return {'policy':'70/15/15 deterministic source-frame components; exact crop duplicates link frames',
        'groups':len(groups),'frame_counts':dict(Counter(assigned[find(f)] for f in parent)),
        'sample_counts':dict(Counter(s['split'] for s in samples))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=ROOT/'data/external/mvtec_screws/source')
    args=parser.parse_args();source=args.source;provenance=json.loads((source.parent/'source.json').read_text())
    dataset=json.loads((source/'mvtec_screws.json').read_text());categories={c['id']:c for c in dataset['categories']}
    originals={i['id']:i for i in dataset['images']};annotations=defaultdict(list)
    for a in dataset['annotations']:annotations[a['image_id']].append(a)
    folder=ROOT/'results/product_families/screw';samples=[];frame_hashes={}
    for image_id,info in sorted(originals.items()):
        image_path=source/info['file_name']
        if not image_path.exists():image_path=source/'images'/Path(info['file_name']).name
        with Image.open(image_path) as original:
            image=original.convert('RGB');frame_hash=pixel_hash(image)
            frame_hashes[str(image_id)]=frame_hash
            for annotation in annotations[image_id]:
                category=int(annotation['category_id'])
                if category not in SCREW_IDS:continue
                crop=aligned_crop(image,annotation['bbox']);pixels=pixel_hash(crop)
                identity=f"{provenance['sha256']}:{image_id}:{annotation['id']}"
                key=hashlib.sha256(identity.encode()).hexdigest();relative=f'product_families/screw/mvtec_screws_v1/crops/{key}.png'
                target=ROOT/'results'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                crop.save(target,compress_level=6)
                samples.append({'key':key,'path':relative,'category_id':category,'category_name':ALIASES[category],
                    'source_category':categories[category],'source_frame':frame_hash,'source_image_id':image_id,
                    'source_file':info['file_name'],'source_annotation_id':annotation['id'],'source_bbox':annotation['bbox'],
                    'pixel_hash':pixels,'file_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                    'quality_source':'not_provided','localization':'publisher_annotation_assisted','category_conflict':False})
        if len(frame_hashes)%64==0:print(json.dumps({'source_frames_prepared':len(frame_hashes),'candidate_crops':len(samples)}),flush=True)
    split=group_splits(samples)
    by_hash=defaultdict(set)
    for sample in samples:by_hash[sample['pixel_hash']].add(sample['category_id'])
    for sample in samples:sample['category_conflict']=len(by_hash[sample['pixel_hash']])>1
    types=[{'category_id':c,'name':ALIASES[c],'publisher_category':categories[c],
        'count':sum(s['category_id']==c for s in samples),'split_counts':dict(Counter(s['split'] for s in samples if s['category_id']==c))}
        for c in sorted(SCREW_IDS)]
    # Thumbnail order interleaves types and makes train candidates convenient to inspect.
    samples.sort(key=lambda s:({'train':0,'validation':1,'test':2}[s['split']],s['source_image_id'],s['source_annotation_id']))
    provided={}
    hashes_by_name={originals[int(key)]['file_name']:value for key,value in frame_hashes.items()}
    for part in ('train','val','test'):
        data=json.loads((source/f'mvtec_screws_{part}.json').read_text())
        # Publisher resets image/annotation IDs within each split. Match file names,
        # never those local IDs, to the full dataset's content hashes.
        provided[part]={'frames':{hashes_by_name[i['file_name']] for i in data['images']}}
    audit={'publisher_total_images':len(originals),'publisher_total_objects':len(dataset['annotations']),
        'readme_reported_objects':4426,'annotation_count_note':'Use actual JSON record count; README states 4426. No silent deletion to match documentation.',
        'publisher_categories':list(categories.values()),'selected_screw_categories':sorted(SCREW_IDS),
        'excluded_nut_categories':sorted(set(categories)-SCREW_IDS),'partition':split,
        'publisher_split_frame_overlaps':{f'{a}/{b}':len(provided[a]['frames']&provided[b]['frames']) for a,b in (('train','val'),('train','test'),('val','test'))},
        'category_conflicts':sum(s['category_conflict'] for s in samples),'duplicate_crops':len(samples)-len(by_hash),
        'quality_labels_provided':False,'model_changed':False,'physical_workpiece_ids_available':False}
    manifest={'schema':1,'dataset':provenance,'summary':{'types':len(types),'samples':len(samples),'source_frames':len({s['source_frame'] for s in samples}),
        'quality_verified':0,'reference_approved':0},'types':types,'samples':samples,'audit':audit,
        'alias_source':'Tutorial and crop-visualization descriptive aliases; display names, not verified specifications'}
    old=folder/'appearance.json'
    if old.exists():
        previous=json.loads(old.read_text())
        if {s['key']:s['pixel_hash'] for s in previous['samples']}!={s['key']:s['pixel_hash'] for s in samples}:raise ValueError('Existing candidate manifest differs; create an explicit new version')
    write(old,manifest);write(folder/'mvtec_screws_v1/audit.json',audit)
    # Contact sheet for geometry QA; all examples remain quality-unverified.
    sheet=Image.new('RGB',(900,len(types)*160),'#eef1f4');draw=ImageDraw.Draw(sheet)
    for row,t in enumerate(types):
        examples=[s for s in samples if s['category_id']==t['category_id']][:6]
        for col,sample in enumerate(examples):
            with Image.open(ROOT/'results'/sample['path']) as image:
                image.thumbnail((130,125));sheet.paste(image,(col*150+(150-image.width)//2,row*160+23))
            draw.text((col*150+6,row*160+3),f"class {t['category_id']} / {sample['source_annotation_id']}",fill='#142c40')
    sheet.save(folder/'mvtec_screws_v1/alignment-contact.jpg',quality=93)
    print(json.dumps({'summary':manifest['summary'],'audit':audit},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
