"""Shared offline/online feature extraction; no labels or cloud calls here."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PATCHCORE_WEIGHTS = ROOT / "weights/torch_hub/checkpoints/wide_resnet50_2-9ba9bcbe.pth"


def local_crops(image):
    """Four overlapping 60%-width/height crops, retaining original fine detail."""
    w, h = image.size
    cw, ch = max(1, round(w * .6)), max(1, round(h * .6))
    return [image.crop((x, y, x + cw, y + ch))
            for y in (0, h - ch) for x in (0, w - cw)]


def cosine_anomalies(features, bank):
    return np.maximum(0, 1.0 - (features @ bank.T).max(axis=1))


def euclidean_anomalies(features, bank):
    distances = ((features * features).sum(1, keepdims=True)
                 + (bank * bank).sum(1)[None, :] - 2 * features @ bank.T)
    return np.sqrt(np.maximum(0, distances.min(axis=1)))


def representative_bank(features, size=1024, seed=42):
    """Approximate greedy k-center in a seeded 24-D projection; retain raw vectors.

    The caller first samples patches uniformly per normal image. This makes the
    practical CPU implementation explicit rather than claiming exact PatchCore.
    """
    x = np.asarray(features, dtype=np.float32)
    if len(x) <= size:
        return x.copy()
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(x.shape[1], 24)).astype(np.float32) / np.sqrt(24)
    projected = x @ projection
    current = int(rng.integers(len(x)))
    distances = np.full(len(x), np.inf, dtype=np.float32)
    indices = []
    for _ in range(size):
        indices.append(current)
        delta = projected - projected[current]
        distances = np.minimum(distances, np.einsum("ij,ij->i", delta, delta))
        distances[indices] = -1
        current = int(distances.argmax())
    return x[indices].copy()


class PatchCoreFeatures:
    """WideResNet50 layer2+layer3 local features, pooled to a 28x28 grid.

    This project uses unweighted nearest-neighbor scores and an approximate
    coreset; it is a PatchCore-style baseline, not a full official reproduction.
    """
    def __init__(self):
        from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
        if not PATCHCORE_WEIGHTS.is_file():
            raise FileNotFoundError(f"Missing official backbone weights: {PATCHCORE_WEIGHTS}")
        self.model = wide_resnet50_2(weights=None).eval()
        self.model.load_state_dict(torch.load(PATCHCORE_WEIGHTS, map_location="cpu", weights_only=True))
        self.preprocess = Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms()

    @torch.no_grad()
    def extract_batch(self, images):
        m = self.model
        x = torch.stack([self.preprocess(im.convert("RGB")) for im in images])
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        a = m.layer2(m.layer1(x))
        b = m.layer3(a)
        a = F.avg_pool2d(a, 3, 1, 1)
        b = F.avg_pool2d(b, 3, 1, 1)
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([a, b], dim=1).permute(0, 2, 3, 1).flatten(1, 2).numpy()


def binary_probability(features, params, prefix):
    x = np.asarray(features, dtype=np.float64).reshape(-1, len(params[prefix + "_mean"]))
    z = ((x - params[prefix + "_mean"]) / params[prefix + "_scale"])
    logits = z @ params[prefix + "_coef"].T + params[prefix + "_intercept"]
    return 1 / (1 + np.exp(-np.clip(logits[:, 0], -700, 700)))


def export_binary(pipeline, prefix):
    sc, clf = pipeline.named_steps["sc"], pipeline.named_steps["clf"]
    if clf.classes_.tolist() != [0, 1]:
        raise ValueError("Binary artifact must have classes [0, 1]")
    return {prefix + "_mean": sc.mean_, prefix + "_scale": sc.scale_,
            prefix + "_coef": clf.coef_, prefix + "_intercept": clf.intercept_}


class OptimizedPerception:
    def __init__(self, metadata_path, patchcore_factory=PatchCoreFeatures):
        import json
        path = Path(metadata_path)
        self.metadata = json.loads(path.read_text(encoding="utf-8"))
        with np.load(path.parent / self.metadata["arrays"], allow_pickle=False) as arrays:
            self.params = {key: arrays[key].copy() for key in arrays.files}
        self.patchcore_factory = patchcore_factory
        self.patchcore = None

    def apply(self, detector, image, result):
        mode = self.metadata["mode"]
        baseline = float(result["score"])
        global_prob = float(binary_probability(result["image_features"], self.params, "global")[0])
        crop_score = patchcore_score = 0.0
        heatmap = result["heatmap"]
        if mode in ("multiscale", "fusion"):
            crops = local_crops(image)
            patches, _ = detector.extract_batch(crops)
            crop_score = float(cosine_anomalies(patches.reshape(-1, patches.shape[-1]),
                                               self.params["crop_bank"]).max())
        if mode in ("patchcore", "fusion"):
            if self.patchcore is None:
                self.patchcore = self.patchcore_factory()
            patches = self.patchcore.extract_batch([image])[0]
            anomaly = euclidean_anomalies(patches, self.params["patchcore_bank"])
            patchcore_score = float(anomaly.max())
            heatmap = detector._overlay(image, anomaly.reshape(28, 28))
        if mode == "global_binary":
            score = global_prob
        elif mode == "fast_fusion":
            score = float(binary_probability([baseline, global_prob], self.params, "fusion")[0])
        elif mode == "fusion":
            score = float(binary_probability([baseline, crop_score, patchcore_score, global_prob],
                                             self.params, "fusion")[0])
        elif mode == "multiscale":
            score = max(baseline, crop_score)
        elif mode == "patchcore":
            score = patchcore_score
        else:
            raise ValueError(f"Unknown optimized mode: {mode}")
        return {**result, "score": score, "heatmap": heatmap, "baseline_score": baseline,
                "method": self.metadata["name"], "global_defect_probability": global_prob}
