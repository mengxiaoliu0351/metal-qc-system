"""Verified image-derivative identities and evaluation exclusion.

Approximate matching is an offline intake operation. Online code consumes only
the reviewed alias registry; it never infers a quality label from similarity.
"""
from pathlib import Path
import re

from products.families import read

REGISTRY = 'data_intake/known_image_derivatives.json'
HEX = re.compile(r'[a-f0-9]{64}')


def aliases(results):
    path = Path(results)/REGISTRY
    if not path.exists(): return {}
    state = read(path)
    if state.get('schema') != 1 or not isinstance(state.get('aliases'), dict):
        raise ValueError('来源图像对应登记无效，暂停数据使用')
    values = {}
    for key, value in state['aliases'].items():
        target = value.get('canonical_pixel_hash') if isinstance(value, dict) else None
        if not isinstance(key, str) or not HEX.fullmatch(key) or not isinstance(target, str) or not HEX.fullmatch(target):
            raise ValueError('来源图像对应标识无效，暂停数据使用')
        values[key] = target
    for key in values: canonical(key, values)
    return values


def canonical(fingerprint, mapping):
    seen = set()
    while fingerprint in mapping:
        if fingerprint in seen or len(seen) >= 32:
            raise ValueError('来源图像对应存在循环或链条过长，暂停数据使用')
        seen.add(fingerprint); fingerprint = mapping[fingerprint]
    return fingerprint


def reserved_hashes(results):
    path = Path(results)/'evaluation_reservations.json'
    state = read(path) if path.exists() else {}
    protected = set(state.get('global', {}))
    # Changing the product name must not turn another product's sealed image
    # into independent training data.
    for group in state.get('products', {}).values(): protected.update(group)
    mapping = aliases(results)
    return protected | {key for key in mapping if canonical(key, mapping) in protected}
