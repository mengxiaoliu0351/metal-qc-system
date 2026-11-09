"""Verify the original archive and retain a fixed, specimen-grouped small pilot."""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tarfile

import numpy as np
from PIL import Image

from experiments.download_csem_screw import DIRECTORY, SIZE, MD5
from products.families import read, write
from review.workflow import image_hash, file_hash

ROOT = Path(__file__).resolve().parents[2]
LIGHTS = ('501', '507', '901')


def safe_member(name):
    value = PurePosixPath(name)
    if value.is_absolute() or PureWindowsPath(name).drive or '..' in value.parts or '\\' in name:
        raise ValueError('Unsafe original archive path')
    return value


def specimen_label(partition, specimen, mask):
    # Some annotated numeric IDs are defective too (e.g. 203). Only the actual
    # author's mask supplies quality truth; never infer normality from a name.
    if partition not in ('Train', 'Test'): return None
    if not re.fullmatch(r'(?:defect_)?[0-9]{3}', specimen): raise ValueError('Unknown annotated specimen convention')
    if mask is None: raise ValueError('Annotated specimen lacks a source quality mask')
    return 'defect' if np.any(mask > 0) else 'normal'


def extract(directory):
    archive = directory/'screw.tar.gz'; output = directory/'pilot_source'; manifest_path = directory/'pilot_manifest.json'
    if manifest_path.exists(): raise ValueError('CSEM intake already frozen; inspect existing manifest')
    md5 = hashlib.md5(); sha = hashlib.sha256(); size = 0
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            md5.update(block); sha.update(block); size += len(block)
    if size != SIZE or md5.hexdigest() != MD5: raise ValueError('Original author archive checksum mismatch')
    output.mkdir(exist_ok=True); groups = defaultdict(dict); unannotated = set(); members = 0; total = 0
    test_reservations = {}
    with tarfile.open(archive, 'r|gz') as bundle:
        for item in bundle:
            parts = safe_member(item.name).parts; members += 1; total += item.size
            if members > 200000 or total > 32*1024**3 or item.issym() or item.islnk(): raise ValueError('Unexpected archive contents')
            if not item.isfile(): continue
            if len(parts) >= 3 and parts[0] == 'Screw' and parts[1] == 'Unannotated': unannotated.add(parts[2])
            if len(parts) != 4 or parts[0] != 'Screw' or parts[1] not in ('Train', 'Test'): continue
            _, partition, specimen, filename = parts
            if not re.fullmatch(r'(?:defect_)?[0-9]{3}', specimen): raise ValueError('Unexpected specimen ID')
            image_match = re.fullmatch(re.escape('screw_'+specimen+'_')+r'([0-9]{3})\.png', filename)
            kind = image_match[1] if image_match else 'mask' if filename == f'screw_{specimen}_mask.png' else 'foreground' if filename == f'screw_{specimen}_segmentation.png' else None
            # All 108 photos of a held-out physical part stay excluded, even
            # when this pilot retains just three of them on disk.
            if partition == 'Test' and image_match and kind not in LIGHTS:
                import io
                if item.size > 2*1024**2: raise ValueError('Unexpected held-out image size')
                with Image.open(io.BytesIO(bundle.extractfile(item).read())) as image:
                    rgb = image.convert('RGB')
                    fingerprint = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                test_reservations[fingerprint] = {'specimen_id': specimen, 'illumination': kind}
            if kind not in LIGHTS+('mask', 'foreground'): continue
            if item.size > 2*1024**2: raise ValueError('Unexpected image member size')
            target = output/partition/specimen/filename
            if not target.resolve().is_relative_to(output.resolve()): raise ValueError('Extraction escaped pilot directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            content = bundle.extractfile(item).read()
            if len(content) != item.size: raise ValueError('Truncated image member')
            if kind in groups[partition, specimen]: raise ValueError('Duplicate selected archive member')
            target.write_bytes(content); groups[partition, specimen][kind] = target
    specimens = []; samples = []; identities = defaultdict(list)
    for (partition, specimen), paths in sorted(groups.items()):
        if any(light not in paths for light in LIGHTS) or 'foreground' not in paths: raise ValueError('Missing fixed illumination or foreground mask')
        mask = None
        if 'mask' in paths:
            with Image.open(paths['mask']) as im:
                mask = np.asarray(im)
                if mask.ndim == 3: mask = mask[..., 0]
                if mask.shape != (512, 512): raise ValueError('Unexpected defect mask dimensions')
        label = specimen_label(partition, specimen, mask)
        specimens.append({'specimen_id': specimen, 'publisher_partition': partition, 'label': label,
                          'source_mask_present': 'mask' in paths, 'mask_nonzero_pixels': int(np.sum(mask > 0)) if mask is not None else 0,
                          'mask_path': paths['mask'].relative_to(ROOT).as_posix() if 'mask' in paths else None})
        for light in LIGHTS:
            path = paths[light]
            with Image.open(path) as image:
                if image.size != (512, 512): raise ValueError('Unexpected source image dimensions')
            fingerprint = image_hash(path); identities[fingerprint].append((partition, specimen))
            if partition == 'Test': test_reservations[fingerprint] = {'specimen_id': specimen, 'illumination': light}
            samples.append({'specimen_id': specimen, 'publisher_partition': partition, 'label': label, 'illumination': light,
                            'path': path.relative_to(ROOT).as_posix(), 'file_sha256': file_hash(path), 'pixel_hash': fingerprint})
    if len({s['specimen_id'] for s in specimens}) != len(specimens): raise ValueError('Specimen crossed publisher partitions')
    if any(len(set(values)) > 1 for values in identities.values()): raise ValueError('Exact source image crossed specimen identities')
    known = set()
    for name in ('head-manifest.json', 'body-manifest.json'):
        known.update(s['pixel_hash'] for s in read(ROOT/'data/external/m2_v2'/name)['images'])
    known.update(image_hash(path) for path in (ROOT/'data/mvtec/screw').rglob('*.png'))
    overlap = (set(identities) | set(test_reservations)) & known
    if overlap: raise ValueError('New source overlaps decoded pixels of existing screw datasets')
    counts = Counter((s['publisher_partition'], s['label']) for s in specimens)
    if len(specimens) != 70 or sum(s['label'] == 'normal' for s in specimens) != 35 or sum(s['label'] == 'defect' for s in specimens) != 35:
        raise ValueError('Actual labelled counts differ from author metadata; stop for review')
    if sum(s['publisher_partition'] == 'Train' for s in specimens) != 32: raise ValueError('Official train count changed')
    write(manifest_path, {'dataset': 'CSEM-MISD v0.2.0 screws, fixed three-light pilot', 'doi': '10.5281/zenodo.7410513',
          'source': 'https://zenodo.org/records/7410513', 'license': 'CC BY-NC-ND 4.0',
          'archive_md5': md5.hexdigest(), 'archive_sha256': sha.hexdigest(), 'lights': list(LIGHTS),
          'label_basis': 'nonempty author defect mask = defect; empty author defect mask = normal; mask required, filename not used as quality; Unannotated excluded',
          'specimens': specimens, 'images': samples,
          'test_specimen_image_hashes': test_reservations, 'intake_implementation_sha256': file_hash(Path(__file__)),
          'existing_source_exact_identity_audit': {'known_images': len(known), 'overlap': len(overlap),
              'scope': 'decoded pixel identities against MVTec AD screws and both M2 manifests; not an exhaustive near-duplicate proof'},
          'summary': {'physical_specimens': len(specimens), 'selected_images': len(samples), 'unannotated_specimens_excluded': len(unannotated),
                      'partitions': {part: {label: counts[part, label] for label in ('normal', 'defect')} for part in ('Train', 'Test')}},
          'web_image_publication': False, 'automatic_archival': False, 'quality_label_permission_for_live_models': False})
    print(read(manifest_path)['summary'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args(); extract(args.directory.resolve())


if __name__ == '__main__': main()
