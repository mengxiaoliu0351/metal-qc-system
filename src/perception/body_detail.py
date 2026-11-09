"""Offline BODY detail baseline; separate from the frozen production preprocessing.

The proposal joins nearby dark components to avoid selecting a highlight-separated
fragment. It has no localization ground truth and must not be called a detector.
"""
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

VERSION = 'body-detail-components-v1'


def prepare_detail(image):
    image = image.convert('RGB')
    small = ImageOps.contain(image, (512, 512), Image.Resampling.BILINEAR)
    gray = np.asarray(small.convert('L'))
    h, w = gray.shape
    mask = gray < 150
    mask[:int(.15*h)] = False
    mask[int(.75*h):] = False
    mask[:, :int(.2*w)] = False
    mask[:, int(.8*w):] = False
    mask = ndimage.binary_closing(mask, structure=np.ones((7, 3)))
    labelled, _ = ndimage.label(mask)
    areas = np.bincount(labelled.ravel())
    boxes = []
    for index, slices in enumerate(ndimage.find_objects(labelled), 1):
        if slices is None or areas[index] < 20:
            continue
        ys, xs = slices
        bw, bh = xs.stop-xs.start, ys.stop-ys.start
        cx = (xs.start+xs.stop)/(2*w)
        if not (.01 <= bw/w <= .32 and .015 <= bh/h <= .55 and .3 <= cx <= .7):
            continue
        boxes.append((int(areas[index]), (xs.start, ys.start, xs.stop, ys.stop)))
    if not boxes:
        return None, {'version': VERSION, 'proposal_found': False, 'source_box': None}
    _, box = max(boxes, key=lambda item: item[0]*(item[1][3]-item[1][1]))
    # Join vertical fragments only when they overlap horizontally. Broad table
    # components were excluded above; no class label is used in this operation.
    remaining = [candidate for _, candidate in boxes if candidate != box]
    while True:
        additions = []
        for candidate in remaining:
            x1, y1, x2, y2 = box
            a, b, c, d = candidate
            overlap = max(0, min(x2, c)-max(x1, a))
            gap = max(0, max(y1, b)-min(y2, d))
            merged = (min(x1, a), min(y1, b), max(x2, c), max(y2, d))
            if (overlap >= .25*min(x2-x1, c-a) and gap <= .04*h
                    and merged[3]-merged[1] <= .55*h and merged[2]-merged[0] <= .32*w):
                additions.append(candidate)
                box = merged
        if not additions:
            break
        remaining = [candidate for candidate in remaining if candidate not in additions]
    x1, y1, x2, y2 = box
    if y2-y1 < .08*h or not (1.1 <= (y2-y1)/max(x2-x1, 1) <= 12):
        return None, {'version': VERSION, 'proposal_found': False, 'source_box': None}
    mx, my = .15*(x2-x1), .08*(y2-y1)
    source = (max(0, int((x1-mx)*image.width/w)), max(0, int((y1-my)*image.height/h)),
              min(image.width, int(np.ceil((x2+mx)*image.width/w))),
              min(image.height, int(np.ceil((y2+my)*image.height/h))))
    crop = image.crop(source)
    detail = ImageOps.pad(crop, (96, 256), Image.Resampling.LANCZOS, color=(128, 128, 128))
    detail = ImageOps.autocontrast(detail.convert('L'), cutoff=1)
    return detail, {'version': VERSION, 'proposal_found': True, 'source_box': list(source),
                    'scope': 'unlabelled BODY proposal; not evaluated localization or arbitrary view recognition'}


def detail_features(image):
    from skimage.feature import hog
    detail, metadata = prepare_detail(image)
    if detail is None:
        return None, metadata
    values = hog(np.asarray(detail, dtype=np.float32)/255, orientations=9,
                 pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys',
                 transform_sqrt=True, feature_vector=True)
    return np.asarray(values, dtype=np.float32), metadata
