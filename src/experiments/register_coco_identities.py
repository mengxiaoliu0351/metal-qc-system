"""Install verified intake correspondences, without importing images or labels."""
import argparse
from pathlib import Path

from experiments.audit_screw_coco import ROOT, MATCH
from products.families import read, write
from review.data_identity import REGISTRY, aliases, canonical
from review.workflow import file_hash, image_hash


def register(folder):
    receipt = folder/'identity_registration.json'
    if receipt.exists(): raise ValueError('Registration already completed; inspect the frozen receipt')
    manifest = read(folder/'manifest.json')
    if manifest['matching_protocol'] != MATCH or file_hash(ROOT/manifest['archive']) != manifest['archive_sha256']:
        raise ValueError('Intake archive or matching protocol changed')
    if file_hash(ROOT/'src/experiments/audit_screw_coco.py') != manifest['implementation_sha256']:
        raise ValueError('Intake implementation changed')
    path = ROOT/'results'/REGISTRY
    state = read(path) if path.exists() else {'schema': 1, 'aliases': {}}
    before = read(path) if path.exists() else None
    added = 0
    for sample in manifest['images']:
        match = sample['correspondence']
        if not match['high_confidence_match']: continue
        if match['rmse'] > MATCH['maximum_rmse'] or match['best_second_ratio'] > MATCH['maximum_best_second_ratio'] or match['phash_distance'] > MATCH['maximum_phash_distance']:
            raise ValueError('Correspondence does not meet frozen thresholds')
        source = match['nearest_reference']; fingerprint = sample['pixel_hash']
        target = (ROOT/source['path']).resolve()
        if not target.is_relative_to((ROOT/'data/mvtec/screw').resolve()) or image_hash(target) != source['pixel_hash']:
            raise ValueError('Canonical image changed or escaped reference directory')
        if fingerprint == source['pixel_hash']: continue
        previous = state['aliases'].get(fingerprint)
        if previous and previous['canonical_pixel_hash'] != source['pixel_hash']:
            raise ValueError('Conflicting canonical correspondence')
        if not previous:
            state['aliases'][fingerprint] = {'canonical_pixel_hash': source['pixel_hash'], 'canonical_path': source['path'],
                'audit': folder.relative_to(ROOT/'results').as_posix(), 'archive_sha256': manifest['archive_sha256'],
                'rmse': match['rmse'], 'phash_distance': match['phash_distance'], 'quality_label_permission': False}
            added += 1
    mapping = {key: value['canonical_pixel_hash'] for key, value in state['aliases'].items()}
    for key in mapping: canonical(key, mapping)
    write(folder/'identity_registry_before.json', before)
    write(path, state); aliases(ROOT/'results')
    write(receipt, {'added': added, 'registry_sha256': file_hash(path), 'quality_labels_imported': 0,
                    'models_modified': False, 'images_imported': 0})
    print({'verified_identities_added': added, 'images_or_labels_imported': 0}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=ROOT/'results/data_intake/gunadarma_screw_v2')
    args = parser.parse_args(); register(args.audit.resolve())


if __name__ == '__main__': main()
