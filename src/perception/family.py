"""One tagged reference bank and shared classifiers for a product-family pilot."""
import hashlib
import json
from pathlib import Path

import numpy as np

from perception.feature_backends import binary_probability


def unit(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all(): raise ValueError("Non-finite family features")
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def conditional_features(features, groups, n_groups):
    x = np.asarray(features, dtype=np.float64).reshape(-1, np.asarray(features).shape[-1])
    groups = np.asarray(groups, dtype=int).reshape(-1)
    if len(x) != len(groups) or np.any(groups < 0) or np.any(groups >= n_groups):
        raise ValueError("Invalid reference group")
    return np.c_[x, np.eye(n_groups)[groups]]


class FamilyModel:
    """Source/view groups are reference conditions, not verified SKU identities.

    Pooled and structured inference never accept a true source label. Known-source
    selection is only used by the explicitly named separate/oracle comparator.
    All modes are advisory; no automatic archival permission is granted here.
    """
    def __init__(self, metadata_path):
        self.path = Path(metadata_path)
        self.meta = json.loads(self.path.read_text(encoding="utf8"))
        array_path = (self.path.parent / self.meta["arrays"]).resolve()
        if array_path.parent != self.path.parent.resolve(): raise ValueError("Invalid family artifact path")
        if hashlib.sha256(array_path.read_bytes()).hexdigest() != self.meta["arrays_sha256"]:
            raise ValueError("Family model checksum mismatch")
        with np.load(array_path, allow_pickle=False) as data:
            self.params = {k: data[k].copy() for k in data.files}
        self.groups = self.meta["groups"]
        self.n_groups = len(self.groups)
        owners = self.params["bank_group"]
        if self.n_groups < 2 or len(owners) != len(self.params["bank"]) or set(owners.tolist()) != set(range(self.n_groups)):
            raise ValueError("Family bank and group metadata do not match")
        self.centroids = unit(self.params["centroids"])

    def route(self, features):
        similarities = unit(np.asarray(features).reshape(-1)) @ self.centroids.T
        group = int(np.argmax(similarities))
        return {"group": group, "reference_group": self.groups[group]["key"],
                "reference_name": self.groups[group]["name"],
                "similarity": float(similarities[group]),
                "in_scope": bool(similarities[group] >= self.meta["scope_min_similarity"][group]),
                "scope": "source/view reference routing, not SKU recognition"}

    def predict(self, features, known_group=None):
        vector = np.asarray(features, dtype=np.float64).reshape(1, -1)
        route = self.route(vector)
        predicted_group = route["group"]
        pooled = float(binary_probability(vector, self.params, "pooled")[0])
        structured = float(binary_probability(conditional_features(vector, [predicted_group], self.n_groups),
                                             self.params, "structured")[0])
        scores = {"pooled": pooled, "structured": structured}
        scores["routed_expert"] = float(binary_probability(vector, self.params, "separate_" + str(predicted_group))[0])
        boundaries = {"pooled": self.meta["thresholds"]["pooled"],
                      "structured": self.meta["thresholds"]["structured"][predicted_group],
                      "routed_expert": self.meta["thresholds"]["separate"][predicted_group]}
        if known_group is not None:
            if isinstance(known_group, bool) or not isinstance(known_group, int) or not 0 <= known_group < self.n_groups:
                raise ValueError("Invalid known reference group")
            scores["separate"] = float(binary_probability(vector, self.params, "separate_" + str(known_group))[0])
            boundaries["separate"] = self.meta["thresholds"]["separate"][known_group]
        return {"routing": route, "scores": scores, "thresholds": boundaries,
                "predictions": {k: "defect" if v >= boundaries[k] else "normal" for k, v in scores.items()},
                "requires_review": True, "automatic_archival": False}

    def patch_scores(self, patches):
        x = np.asarray(patches, dtype=np.float32)
        similarities = x @ self.params["bank"].T
        grouped = np.array([np.maximum(0, 1 - similarities[:, self.params["bank_group"] == group].max(axis=1))
                            for group in range(self.n_groups)])
        return {"grouped": grouped, "pooled": grouped.min(axis=0)}
