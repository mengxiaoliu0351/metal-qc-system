"""Read-only COCO intake audit, including resized/re-encoded known-image matching.

No imported category or annotation activates a quality model. No files from an
untrusted archive are extracted and no source labels or live registries change.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.fft import dctn

from products.families import read, write
from review.workflow import file_hash

ROOT = Path(__file__).resolve().parents[2]
LABELS = {'Screw-OK': 'normal', 'Screw-Defect': 'defect'}
MATCH = {'thumbnail_size': 64, 'maximum_rmse': .01, 'maximum_best_second_ratio': .25,
         'maximum_phash_distance': 4, 'scope': 'high-confidence resized/re-encoded image correspondence, not physical identity'}


def safe_name(name):
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    if (not name or '\\' in name or posix.is_absolute() or windows.drive
            or '..' in posix.parts or '\x00' in name):
        raise ValueError('Unsafe archive or COCO filename')
    return posix


def validate_archive(bundle):
    infos = bundle.infolist()
    if len(infos) > 20000 or sum(i.file_size for i in infos) > 4*1024**3:
        raise ValueError('Archive exceeds intake limits')
    names = set()
    for item in infos:
        safe_name(item.filename)
        if item.filename in names or stat.S_ISLNK(item.external_attr >> 16):
            raise ValueError('Duplicate archive member or symbolic link')
        if item.file_size > 32*1024**2:
            raise ValueError('Archive member exceeds intake limit')
        names.add(item.filename)


def fingerprint(image):
    image = image.convert('RGB')
    pixel = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
    gray = image.convert('L')
    thumbnail = np.asarray(gray.resize((64, 64), Image.Resampling.LANCZOS), np.float32).ravel()/255
    low = dctn(np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), float), norm='ortho')[:8, :8]
    bits = low.ravel() > np.median(low.ravel()[1:])
    return pixel, thumbnail, bits


def load_coco(bundle):
    import json
    validate_archive(bundle)
    samples, vectors, hashes = [], [], []
    for member in sorted(bundle.namelist()):
        if not member.endswith('_annotations.coco.json'):
            continue
        doc = json.loads(bundle.read(member)); parent = PurePosixPath(member).parent
        categories = {c['id']: c['name'] for c in doc['categories']}
        if len(categories) != len(doc['categories']):
            raise ValueError('Duplicate category ID')
        images = {s['id']: s for s in doc['images']}
        if len(images) != len(doc['images']):
            raise ValueError('Duplicate image ID')
        annotations = defaultdict(list)
        for ann in doc['annotations']:
            if ann['image_id'] not in images or ann['category_id'] not in categories:
                raise ValueError('Annotation refers to missing image or category')
            annotations[ann['image_id']].append(ann)
        for metadata in sorted(doc['images'], key=lambda item: item['file_name']):
            relative = safe_name(metadata['file_name']); path = str(parent/relative)
            raw = bundle.read(path)
            with Image.open(io.BytesIO(raw)) as original:
                if original.width*original.height > 20_000_000 or min(original.size) < 16:
                    raise ValueError('Image dimensions exceed intake limits')
                image = original.convert('RGB')
                if image.size != (metadata['width'], metadata['height']):
                    raise ValueError('COCO dimensions do not match decoded image')
                pixel, vector, perceptual = fingerprint(image)
            anns = annotations[metadata['id']]
            names = [categories[a['category_id']] for a in anns]
            flags = []; boxes = []
            for ann in anns:
                box = ann.get('bbox', [])
                if (len(box) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
                        or min(box[2:]) <= 0 or min(box[:2]) < 0
                        or box[0]+box[2] > metadata['width']+1 or box[1]+box[3] > metadata['height']+1):
                    flags.append('invalid_bbox'); continue
                boxes.append({'xywh': box, 'category': categories[ann['category_id']],
                              'area_fraction': box[2]*box[3]/(metadata['width']*metadata['height'])})
            if len(anns) != 1: flags.append('requires_instance_semantics_review')
            if any(name not in LABELS for name in names): flags.append('unmapped_category')
            label = LABELS.get(names[0]) if len(names) == 1 else None
            samples.append({'member': path, 'source_partition': parent.name, 'source_image_name': metadata.get('extra', {}).get('name'),
                            'source_label': label, 'source_categories': names, 'boxes': boxes, 'flags': sorted(set(flags)),
                            'pixel_hash': pixel, 'file_sha256': hashlib.sha256(raw).hexdigest(),
                            'width': metadata['width'], 'height': metadata['height'],
                            'physical_identity': None, 'role': 'intake_quarantine', 'approved_for_training': False})
            vectors.append(vector); hashes.append(perceptual)
    if not samples:
        raise ValueError('No COCO images found')
    return samples, np.asarray(vectors), np.asarray(hashes)


def load_references(root):
    rows = []; vectors = []; hashes = []
    for path in sorted(root.rglob('*.png')):
        with Image.open(path) as image:
            pixel, vector, perceptual = fingerprint(image)
        rows.append({'path': path.relative_to(ROOT).as_posix(), 'pixel_hash': pixel,
                     'source_partition': path.parent.parent.name,
                     'label': 'normal' if path.parent.name == 'good' else 'defect',
                     'defect_category': path.parent.name})
        vectors.append(vector); hashes.append(perceptual)
    if len(rows) < 2: raise ValueError('Need at least two known reference images')
    return rows, np.asarray(vectors), np.asarray(hashes)


def correspondences(vectors, hashes, references, ref_vectors, ref_hashes):
    # Float64 dot products keep tiny recompression distances numerically stable.
    a = np.asarray(vectors, float); b = np.asarray(ref_vectors, float)
    squared = np.maximum(0, np.mean(a*a, axis=1)[:, None]+np.mean(b*b, axis=1)[None, :]-2*(a@b.T)/a.shape[1])
    order = np.argsort(squared, axis=1)[:, :2]; result = []
    for i, (best, second) in enumerate(order):
        rmse = float(np.sqrt(squared[i, best])); next_rmse = float(np.sqrt(squared[i, second]))
        ratio = rmse/next_rmse if next_rmse > 1e-12 else 1.0
        distance = int(np.sum(hashes[i] != ref_hashes[best]))
        accepted = rmse <= MATCH['maximum_rmse'] and ratio <= MATCH['maximum_best_second_ratio'] and distance <= MATCH['maximum_phash_distance']
        result.append({'high_confidence_match': bool(accepted), 'rmse': rmse, 'second_rmse': next_rmse,
                       'best_second_ratio': ratio, 'phash_distance': distance, 'nearest_reference': references[best]})
    return result


def audit(archive, reference, output):
    if (output/'manifest.json').exists(): raise ValueError('Intake audit is frozen; do not overwrite')
    protected = [ROOT/'results/perception/banks/products.json', ROOT/'results/evaluation_reservations.json',
                 ROOT/'results/product_families/screw/operational_router.json', ROOT/'results/product_families/screw/body_quality_candidate.json']
    before = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in protected}
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        samples, vectors, hashes = load_coco(bundle)
        references, rv, rh = load_references(reference)
        matches = correspondences(vectors, hashes, references, rv, rh)
        global_reserved = read(ROOT/'results/evaluation_reservations.json').get('global', {})
        split = read(ROOT/'results/experiments/perception_upgrade_v1/screw_split.json')
        roles = defaultdict(list)
        for role, paths in split.items():
            if not isinstance(paths, list): continue
            for path in paths:
                if isinstance(path, str): roles[path.replace('\\', '/')].append(role)
        counts = Counter(); aliases = []
        for sample, match in zip(samples, matches):
            sample['correspondence'] = match
            if match['high_confidence_match']:
                origin = match['nearest_reference']; match['existing_quality_roles'] = roles[origin['path']]
                match['source_is_globally_reserved'] = origin['pixel_hash'] in global_reserved
                match['source_label_agrees'] = sample['source_label'] == origin['label']
                sample['flags'].append('known_resized_or_reencoded_image')
                if not match['source_label_agrees']: sample['flags'].append('label_conflicts_with_known_source')
                counts['known_correspondences'] += 1
                if match['source_is_globally_reserved']:
                    aliases.append({'pixel_hash': sample['pixel_hash'], 'canonical_pixel_hash': origin['pixel_hash'],
                                    'canonical_path': origin['path'], 'member': sample['member']})
        identities = defaultdict(list)
        for s in samples: identities[s['pixel_hash']].append(s)
        duplicates = [group for group in identities.values() if len(group) > 1]
        partition_counts = {part: dict(Counter(s['source_label'] or 'unmapped' for s in samples if s['source_partition'] == part))
                            for part in sorted({s['source_partition'] for s in samples})}
        boxes = {label: [b['area_fraction'] for s in samples if s['source_label'] == label for b in s['boxes']] for label in ('normal', 'defect')}
        blockers = []
        if any(not {'normal', 'defect'} <= {s['source_label'] for s in samples if s['source_partition'] == part}
               for part in ('valid', 'test')): blockers.append('publisher evaluation split lacks one quality class')
        if counts['known_correspondences']: blockers.append('overlap with existing MVTec images; not a new independent source')
        if any(s['flags'] and any(f != 'known_resized_or_reencoded_image' for f in s['flags']) for s in samples): blockers.append('annotation or source-label issues require review')
        summary = {'images': len(samples), 'source_partitions': partition_counts,
                   'high_confidence_known_correspondences': counts['known_correspondences'],
                   'canonical_known_sources': dict(Counter(m['nearest_reference']['source_partition'] for m in matches if m['high_confidence_match'])),
                   'globally_reserved_derivative_images': len(aliases), 'internal_exact_duplicate_groups': len(duplicates),
                   'canonical_label_conflicts': sum('label_conflicts_with_known_source' in s['flags'] for s in samples),
                   'median_box_area_fraction': {label: float(np.median(v)) if v else None for label, v in boxes.items()},
                   'independent_quality_evaluation_ready': False, 'automatic_archival': False, 'blockers': blockers}
        manifest = {'dataset': 'Gunadarma University Object Detection Screw v2 COCO export',
                    'source': 'https://universe.roboflow.com/gunadarma-university-oo4uc/object-detection-screw/dataset/2',
                    'archive': archive.relative_to(ROOT).as_posix(), 'archive_sha256': file_hash(archive),
                    'license_claim': 'Public Domain in bundled README.dataset.txt; underlying MVTec provenance must also be reviewed',
                    'label_basis': 'publisher COCO category, not human confirmation',
                    'physical_identity': 'not provided', 'matching_protocol': MATCH,
                    'implementation_sha256': file_hash(Path(__file__)), 'summary': summary, 'images': samples}
        write(output/'manifest.json', manifest)
        write(output/'reservation_alias_proposals.json', {'applied': False, 'aliases': aliases,
              'scope': 'proposed derivatives of globally reserved originals; do not import as training'})
        canvas = Image.new('RGB', (1440, 780), 'white'); draw = ImageDraw.Draw(canvas)
        selected = sorted([s for s in samples if s['correspondence']['high_confidence_match']],
                          key=lambda s: s['correspondence']['rmse'], reverse=True)[:12]
        for i, sample in enumerate(selected):
            x, y = (i % 4)*360, (i//4)*260
            with Image.open(io.BytesIO(bundle.read(sample['member']))) as imported:
                canvas.paste(imported.convert('RGB').resize((170, 220)), (x, y+25))
            source = sample['correspondence']['nearest_reference']
            with Image.open(ROOT/source['path']) as original:
                canvas.paste(original.convert('RGB').resize((170, 220)), (x+180, y+25))
            draw.text((x+2, y+2), f"import / existing {source['source_partition']}/{source['defect_category']}/{Path(source['path']).name}", fill='black')
            draw.text((x+2, y+245), f"RMSE={sample['correspondence']['rmse']:.5f}", fill='black')
        canvas.save(output/'correspondence_contact.png')
    after = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in protected}
    if before != after: raise ValueError('Protected production files changed during intake')
    write(output/'preservation.json', {'unchanged': True, 'files': before})
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT/'data/external/Object Detection Screw.v2i.coco.zip')
    parser.add_argument('--reference', type=Path, default=ROOT/'data/mvtec/screw')
    parser.add_argument('--output', type=Path, default=ROOT/'results/data_intake/gunadarma_screw_v2')
    args = parser.parse_args(); audit(args.archive.resolve(), args.reference.resolve(), args.output.resolve())


if __name__ == '__main__': main()
