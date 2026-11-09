"""Select original-colour detail views using CLIP's actual input geometry."""
import math

import numpy as np


def clip_source_box(image, preprocess):
    """Map the supported Resize(short edge) + CenterCrop view back to the source."""
    transforms = getattr(preprocess, "transforms", None)
    if not isinstance(transforms, (list, tuple)):
        return None
    spatial = [t for t in transforms if type(t).__name__ in ("Resize", "CenterCrop")]
    if len(spatial) != 2 or [type(t).__name__ for t in spatial] != ["Resize", "CenterCrop"]:
        return None
    size = spatial[0].size
    if isinstance(size, (tuple, list)) and len(size) == 1:
        size = size[0]
    if not isinstance(size, int) or getattr(spatial[0], "max_size", None) is not None:
        return None
    width, height = image.size
    resized_w = size if width <= height else int(size * width / height)
    resized_h = size if height <= width else int(size * height / width)
    crop = spatial[1].size
    crop_h, crop_w = (crop, crop) if isinstance(crop, int) else crop
    if crop_w > resized_w or crop_h > resized_h:
        return None
    left = round((resized_w - crop_w) / 2)
    top = round((resized_h - crop_h) / 2)
    return (left * width / resized_w, top * height / resized_h,
            (left + crop_w) * width / resized_w, (top + crop_h) * height / resized_h)


def focus_views(image, anomaly_map, preprocess):
    """Return at most two separated, unmodified source crops and their provenance.

    A high patch response is a location hint, including on normal images. Unknown
    preprocessing, flat maps and invalid maps safely produce only the full view.
    """
    metadata = {"version": "focus-v2", "source_size": list(image.size), "boxes": []}
    if anomaly_map is None:
        metadata["reason"] = "missing_map"
        return [], metadata
    scores = np.asarray(anomaly_map, dtype=float)
    if scores.ndim != 2 or min(scores.shape) < 2 or not np.isfinite(scores).all():
        metadata["reason"] = "invalid_map"
        return [], metadata
    if float(np.ptp(scores)) <= 1e-8:
        metadata["reason"] = "flat_map"
        return [], metadata
    view = clip_source_box(image, preprocess)
    if view is None:
        metadata["reason"] = "unsupported_preprocess"
        return [], metadata
    metadata["map_source_box"] = list(view)
    left, top, right, bottom = view
    crop_w, crop_h = (right - left) * .5, (bottom - top) * .5
    height, width = scores.shape
    centres, boxes = [], []
    for index in np.argsort(-scores.ravel(), kind="stable"):
        row, col = np.unravel_index(index, scores.shape)
        x, y = (col + .5) / width, (row + .5) / height
        if any((x - a) ** 2 + (y - b) ** 2 < .35 ** 2 for a, b in centres):
            continue
        cx, cy = left + x * (right - left), top + y * (bottom - top)
        x0 = max(0, min(image.width - crop_w, cx - crop_w / 2))
        y0 = max(0, min(image.height - crop_h, cy - crop_h / 2))
        box = [math.floor(x0), math.floor(y0), math.ceil(x0 + crop_w), math.ceil(y0 + crop_h)]
        boxes.append(box)
        centres.append((x, y))
        if len(boxes) == 2:
            break
    metadata["boxes"] = boxes
    metadata["reason"] = "patch_response_locations"
    return [image.convert("RGB").crop(box) for box in boxes], metadata
