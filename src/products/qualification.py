"""Conservative automatic-action veto using a separate qualification partition."""
import hashlib
import json

import numpy as np

from perception.feature_backends import binary_probability
from review.workflow import file_hash, read


def catalog_profile(results, config):
    product = config["key"]
    folder = results / config["review_benchmark"]
    metadata_path = results / "perception/optimized" / config["perception_artifact"]
    metadata = read(metadata_path)
    arrays_path = metadata_path.parent / metadata["arrays"]
    if metadata["product"] != product or not np.isclose(metadata["threshold"], config["threshold"], rtol=0, atol=1e-10):
        raise ValueError("新产品模型与阈值不配套")
    if file_hash(results / "perception/banks" / config["bank"]) != config["bank_sha256"]:
        raise ValueError("正常记忆库已变化，请重新建库验证")
    split = read(folder / (product + "_split.json"))
    with np.load(folder / (product + "_features.npz"), allow_pickle=False) as data:
        expected = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
        if str(data["signature"]) != expected:
            raise ValueError("新产品分区与特征不匹配")
        index = {str(p): i for i, p in enumerate(data["paths"])}
        calibration = [index[p] for p in split["validation"]]
        qualification = [index[p] for p in split["test"]]
        if set(calibration) & set(qualification):
            raise ValueError("校准与资格验证重叠")
        labels = data["labels"].copy()
        features = data["global_features"].copy()
        centroid = data["normal_global"].mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
        if metadata["mode"] == "normal_memory":
            scores = data["clip_scores"].copy()
        elif metadata["mode"] == "global_binary":
            with np.load(arrays_path, allow_pickle=False) as z:
                params = {k: z[k] for k in z.files}
            scores = binary_probability(features, params, "global")
        else:
            raise ValueError("该新产品模型尚未验证自动处置")
    cs, cy = scores[calibration], labels[calibration]
    both = set(cy.tolist()) == {0, 1}
    auto_low = min(float(config["threshold"]), .5 * float(cs[cy == 1].min())) if both else 0.0
    auto_high = max(float(config["t_high"]), float(np.nextafter(cs[cy == 0].max(), np.inf)))
    qs, qy = scores[qualification], labels[qualification]
    low, high = qs < auto_low, qs > auto_high
    normal_n, defect_n = int(np.sum(qy == 0)), int(np.sum(qy == 1))
    metrics = {"n": len(qy), "normal": normal_n, "defect": defect_n,
               "auto_normal": int(low.sum()), "auto_isolated": int(high.sum()),
               "auto_normal_misses": int(np.sum(low & (qy == 1))),
               "auto_isolation_false_positives": int(np.sum(high & (qy == 0)))}
    similarity = features[calibration] @ centroid / np.maximum(np.linalg.norm(features[calibration], axis=1), 1e-12)
    model_key = hashlib.sha256((json.dumps(config, sort_keys=True) + file_hash(metadata_path) + file_hash(arrays_path)).encode()).hexdigest()
    enough = both and normal_n >= 60 and defect_n >= 60
    return {"product": product, "model_key": model_key, "auto_low": auto_low, "auto_high": auto_high,
            "centroid": centroid.tolist(), "scope_min_similarity": float(max(-1, np.quantile(similarity, .01) - .03)),
            "allow_normal": bool(enough and np.sum(low & (qy == 0)) >= 20 and metrics["auto_normal_misses"] == 0),
            "allow_isolate": bool(enough and np.sum(high & (qy == 1)) >= 20 and metrics["auto_isolation_false_positives"] == 0),
            "regression": metrics, "calibration_n": len(cy),
            "rule": "独立资格分区两类各至少60张、自动区域至少20张且未观察到相应错误；未达标转人工",
            "scope": "来源标签下的外观资格检查；封存测试集未用于资格选择，未知实物身份需披露"}
