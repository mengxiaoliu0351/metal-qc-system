"""Read-only check of the frozen production router on an unseen screw source.

This does not train a router, run a quality model, or create website history.
"""
from collections import Counter
from pathlib import Path
import time

import numpy as np
from PIL import Image

from experiments.csem_screw_pilot import OUTPUT, verify
from experiments.m2_body_quality import ROOT
from perception.appearance import AppearanceMatcher, full_object
from perception.family_router import FamilyRouter
from products.families import read, write
from review.disposition import image_quality
from review.workflow import file_hash


def main():
    destination = OUTPUT/'routing_audit.json'
    if destination.exists():
        raise ValueError('Preserve the original unseen-source routing audit')
    protocol = verify(OUTPUT)
    folder = ROOT/'results/product_families/screw'
    router_pointer = folder/'operational_router.json'
    appearance_pointer = folder/'appearance_active.json'
    router_dir = ROOT/'results'/read(router_pointer)['directory']
    appearance_dir = ROOT/'results'/read(appearance_pointer)['directory']
    model = FamilyRouter(router_dir/'router.json')
    appearance = AppearanceMatcher(appearance_dir/'appearance.json')
    for path, field in ((ROOT/'src/perception/appearance.py', 'preprocessing_sha256'),
                        (ROOT/'src/perception/winclip_patch.py', 'extractor_sha256'),
                        (ROOT/'weights/CLIP-ViT-B-16-open_clip_model.safetensors', 'weights_sha256')):
        if file_hash(path) != model.meta[field]:
            raise ValueError('Audit preprocessing or weights do not match frozen operational router')
    frozen = {str(path.relative_to(ROOT)): file_hash(path) for path in (
        router_pointer, appearance_pointer, model.path, appearance.path,
        ROOT/'src/web/family_inspection.py', ROOT/'src/perception/family_router.py')}
    rows = [s for s in protocol['samples'] if s['role'] == 'test']
    import torch
    from perception.winclip_patch import WinCLIPPatch
    torch.set_num_threads(2)
    detector = WinCLIPPatch(device='cpu')
    results = []; started = time.monotonic()
    for start in range(0, len(rows), 4):
        part = rows[start:start+4]; images = []
        for sample in part:
            path = ROOT/sample['path']
            if file_hash(path) != sample['file_sha256']:
                raise ValueError('Frozen audit source changed')
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        _, vectors = detector.extract_batch([full_object(image) for image in images])
        for sample, image, vector in zip(part, images, vectors):
            routing = model.predict(vector); looks = appearance.predict(vector)
            usable = image_quality(image)['ok']
            dispatched = routing['view_key'] if usable and routing['matched'] and not looks['matched'] else None
            results.append({'specimen_id': sample['specimen_id'], 'illumination': sample['illumination'],
                            'pixel_hash': sample['pixel_hash'], 'input_quality_ok': usable,
                            'routing': routing, 'appearance': looks, 'effective_view': dispatched})
        if start % 24 == 0:
            print({'audited': start+len(part), 'total': len(rows), 'seconds': round(time.monotonic()-started)}, flush=True)
    for relative, expected in frozen.items():
        if file_hash(ROOT/relative) != expected:
            raise ValueError('Operational configuration changed during audit')
    counts = dict(Counter(s['effective_view'] or 'unmatched_review' for s in results))
    write(destination, {'purpose': 'out-of-scope routing audit; no fitting or quality inference',
          'source': 'CSEM-MISD slotted screw heads, absent from operational routing references',
          'independent_specimens': len({s['specimen_id'] for s in results}), 'paired_images': len(results),
          'operational_router_version': model.meta['version'], 'frozen_artifacts': frozen,
          'dispatch_counts': counts, 'all_rejected_by_existing_router': counts.get('unmatched_review', 0) == len(results),
          'production_history_written': False, 'website_quality_models_executed': False, 'rows': results})
    print({'dispatch_counts': counts, 'independent_specimens': 38, 'production_history_written': False}, flush=True)


if __name__ == '__main__':
    main()
