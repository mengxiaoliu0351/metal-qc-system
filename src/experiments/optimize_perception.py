"""Reproducible local benchmark and model selection; no VLM or test-set tuning.

The legacy classifier was fitted on part of MVTec test. It is NEVER used here.
New supervised models use a saved training partition. Validation chooses the
method/threshold. A held-out partition is scored once after selection.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from perception.winclip_patch import WinCLIPPatch
from perception.feature_backends import (PatchCoreFeatures, local_crops, cosine_anomalies,
    euclidean_anomalies, representative_bank, export_binary)

OUT = ROOT / "results/experiments/perception_upgrade_v1"
ARTIFACTS = ROOT / "results/perception/optimized"
SEED = 20261022
KNOWN_DEVELOPMENT = {"metal_nut": {"scratch/001.png", "scratch/000.png", "bent/000.png",
                                      "good/000.png", "bent/005.png"}}
NAMES = {"baseline": "CLIP patch 原始基线", "multiscale": "CLIP 多尺度局部检测",
         "patchcore": "PatchCore 风格局部检测", "global_binary": "CLIP 监督二分类",
         "fast_fusion": "CLIP 异常分数与监督二分类融合", "fusion": "多尺度与 PatchCore 融合"}
COMPLEXITY = {"baseline": 0, "global_binary": 1, "fast_fusion": 2,
              "patchcore": 3, "multiscale": 4, "fusion": 5}


def log(*args):
    print(*args, flush=True)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_manifest(manifest):
    groups = ["train", "validation", "test", "normal_memory", "normal_calibration"]
    for index, name in enumerate(groups):
        files = manifest[name]
        if len(files) != len(set(files)):
            raise ValueError(f"Duplicate files in {name}")
        for other in groups[index + 1:]:
            if set(files) & set(manifest[other]):
                raise ValueError(f"Data leakage between {name} and {other}")
    if not set(manifest.get("known_development", [])) <= set(manifest["train"]):
        raise ValueError("Previously inspected examples must stay in development training")


def make_manifest(product):
    path = OUT / f"{product}_split.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        return manifest
    base = ROOT / "data/mvtec" / product
    labeled = sorted((base / "test").glob("*/*.png"))
    known = KNOWN_DEVELOPMENT.get(product, set())
    development = [p for p in labeled if p.relative_to(base / "test").as_posix() in known]
    eligible = [p for p in labeled if p not in development]
    labels = [p.parent.name for p in eligible]
    train_val, test = train_test_split(eligible, test_size=.25, random_state=SEED, stratify=labels)
    train, validation = train_test_split(train_val, test_size=1/3, random_state=SEED,
                                       stratify=[p.parent.name for p in train_val])
    normal = sorted((base / "train/good").glob("*.png"))
    rng = np.random.default_rng(SEED)
    normal = [normal[i] for i in rng.permutation(len(normal))]
    manifest = {"seed": SEED, "protocol": "labeled pool 50/25/25; inspected demo samples forced into train",
                "train": [str(p.relative_to(ROOT)) for p in train + development],
                "validation": [str(p.relative_to(ROOT)) for p in validation],
                "test": [str(p.relative_to(ROOT)) for p in test],
                "normal_memory": [str(p.relative_to(ROOT)) for p in normal[:80]],
                "normal_calibration": [str(p.relative_to(ROOT)) for p in normal[80:120]],
                "known_development": [str(p.relative_to(ROOT)) for p in development]}
    validate_manifest(manifest)
    save_json(path, manifest)
    return manifest


def open_images(paths):
    images = []
    for p in paths:
        with Image.open(ROOT / p) as image:
            images.append(image.convert("RGB"))
    return images


def extract(product, config, clip, patchcore, manifest, batch_size=8):
    cache = OUT / f"{product}_features.npz"
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["signature"]) == signature:
                log(product, "using complete cached features")
                return {k: data[k].copy() for k in data.files}
    rng = np.random.default_rng(SEED)
    normal_global, crop_pool, patch_pool = [], [], []
    paths = manifest["normal_memory"]
    for start in range(0, len(paths), batch_size):
        images = open_images(paths[start:start + batch_size])
        _, global_features = clip.extract_batch(images)
        normal_global.append(global_features)
        # Four crops are evaluated together in small batches, without resizing the full image first.
        for image in images:
            patches, _ = clip.extract_batch(local_crops(image))
            pool = patches.reshape(-1, patches.shape[-1])
            crop_pool.append(pool[rng.choice(len(pool), 128, replace=False)])
        patches = patchcore.extract_batch(images)
        for features in patches:
            patch_pool.append(features[rng.choice(len(features), 96, replace=False)])
        log(product, "normal memory features", min(start + batch_size, len(paths)), "/", len(paths))
    crop_bank = representative_bank(np.concatenate(crop_pool), size=1024, seed=SEED)
    patch_bank = representative_bank(np.concatenate(patch_pool), size=1024, seed=SEED)
    del crop_pool, patch_pool
    clip_bank = np.load(ROOT / "results/perception/banks" / config["bank"])
    before_bank = clip_bank
    backups = sorted((ROOT / "results/perception/backups").glob("*/repair_manifest.json"))
    for backup in backups:
        candidate = backup.parent / config["bank"]
        if candidate.exists():
            before_bank = np.load(candidate)
            break
    paths = (manifest["train"] + manifest["validation"] + manifest["test"]
             + manifest["normal_calibration"])
    globals_out, scores_out, before_out, local_out, patch_out = [], [], [], [], []
    started = time.perf_counter()
    for start in range(0, len(paths), batch_size):
        images = open_images(paths[start:start + batch_size])
        patches, global_features = clip.extract_batch(images)
        globals_out.append(global_features)
        patch_features = patchcore.extract_batch(images)
        for image, features, pc_features in zip(images, patches, patch_features):
            scores_out.append(float(cosine_anomalies(features, clip_bank).max()))
            before_out.append(float(cosine_anomalies(features, before_bank).max()))
            crop_features, _ = clip.extract_batch(local_crops(image))
            local_out.append(float(cosine_anomalies(crop_features.reshape(-1, features.shape[-1]), crop_bank).max()))
            patch_out.append(float(euclidean_anomalies(pc_features, patch_bank).max()))
        log(product, "evaluation features", min(start + batch_size, len(paths)), "/", len(paths),
            "elapsed", round(time.perf_counter() - started), "s")
    data = {"signature": np.array(signature), "paths": np.array(paths),
            "global_features": np.concatenate(globals_out), "normal_global": np.concatenate(normal_global),
            "clip_scores": np.array(scores_out), "before_scores": np.array(before_out),
            "crop_scores": np.array(local_out), "patchcore_scores": np.array(patch_out),
            "crop_bank": crop_bank, "patchcore_bank": patch_bank,
            "feature_seconds": np.array(time.perf_counter() - started)}
    np.savez_compressed(cache, **data)
    return data


def metrics(y, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "recall": float(tp / max(1, tp + fn)), "false_positive_rate": float(fp / max(1, tn + fp)),
            "auroc": float(roc_auc_score(y, scores)), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def threshold_on_validation(y, scores):
    unique = np.unique(scores)
    candidates = np.r_[np.nextafter(unique[0], -np.inf), (unique[:-1] + unique[1:]) / 2,
                        np.nextafter(unique[-1], np.inf)]
    def objective(t):
        m = metrics(y, scores, t)
        return (m["balanced_accuracy"], m["accuracy"], m["recall"], -m["false_positive_rate"])
    return float(max(candidates, key=objective))


def pipeline(C=1.0):
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(
        C=C, class_weight="balanced", max_iter=2000, random_state=SEED))])


def compare(product, config, data, manifest):
    paths = data["paths"].tolist()
    # No legacy classifier or legacy probabilities enter this experiment.
    y = np.array([int(Path(p).parent.name != "good") for p in paths])
    ntrain, nval, ntest = (len(manifest[k]) for k in ("train", "validation", "test"))
    train = np.arange(ntrain)
    # The old banks used the first 80 sorted train/good images. Some randomly
    # selected calibration normals may overlap those banks; exclude them from
    # EVERY candidate's validation metrics, using file identities, not scores.
    legacy_memory = {str(p.relative_to(ROOT)) for p in
                     sorted((ROOT / "data/mvtec" / product / "train/good").glob("*.png"))[:80]}
    calibration = np.array([i for i in range(ntrain + nval + ntest, len(paths))
                            if paths[i] not in legacy_memory], dtype=int)
    validation = np.r_[np.arange(ntrain, ntrain + nval), calibration]
    test = np.arange(ntrain + nval, ntrain + nval + ntest)
    global_train = np.concatenate([data["global_features"][train], data["normal_global"]])
    global_labels = np.r_[y[train], np.zeros(len(data["normal_global"]), dtype=int)]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_results = {C: float(cross_val_score(pipeline(C), global_train, global_labels,
                     cv=cv, scoring="balanced_accuracy").mean()) for C in (.01, .1, 1.)}
    best_C = max(cv_results, key=cv_results.get)
    classifier = pipeline(best_C).fit(global_train, global_labels)
    global_scores = classifier.predict_proba(data["global_features"])[:, 1]
    # Stacking sees out-of-fold supervised predictions, never in-sample predictions.
    oof = cross_val_predict(pipeline(best_C), global_train, global_labels, cv=cv, method="predict_proba")[:ntrain, 1]
    features = np.column_stack([data["clip_scores"], data["crop_scores"], data["patchcore_scores"], global_scores])
    train_features = features[train].copy()
    train_features[:, -1] = oof
    fusion = pipeline(.1).fit(train_features, y[train])
    fast = pipeline(.1).fit(train_features[:, [0, 3]], y[train])
    score_sets = {"baseline": data["clip_scores"],
                  "multiscale": np.maximum(data["clip_scores"], data["crop_scores"]),
                  "patchcore": data["patchcore_scores"], "global_binary": global_scores,
                  "fast_fusion": fast.predict_proba(features[:, [0, 3]])[:, 1],
                  "fusion": fusion.predict_proba(features)[:, 1]}
    thresholds = {mode: (float(config["threshold"]) if mode == "baseline"
                         else threshold_on_validation(y[validation], scores[validation]))
                  for mode, scores in score_sets.items()}
    validation_results = {mode: metrics(y[validation], scores[validation], thresholds[mode])
                          for mode, scores in score_sets.items()}
    # Selection is frozen BEFORE looking at held-out metrics.
    def rank(mode):
        m = validation_results[mode]
        return (m["balanced_accuracy"], m["accuracy"], m["recall"], -COMPLEXITY[mode])
    selected = max(score_sets, key=rank)
    selection = {"selected": selected, "threshold": thresholds[selected],
                 "validation": validation_results, "global_C": best_C,
                 "cv_training_only": cv_results, "selected_without_test_metrics": True}
    save_json(OUT / f"{product}_selection.json", selection)
    test_results = {mode: metrics(y[test], scores[test], thresholds[mode]) for mode, scores in score_sets.items()}
    all_labeled = np.arange(ntrain + nval + ntest)
    report = {**selection, "test": test_results,
              "before_repair_full_anomaly_only": metrics(y[all_labeled], data["before_scores"][all_labeled], config["threshold"]),
              "clean_full_anomaly_only": metrics(y[all_labeled], data["clip_scores"][all_labeled], config["threshold"]),
              "warning": "Only the held-out partition evaluates supervised/fusion accuracy. Full-pool supervised accuracy is not reported.",
              "split": {k: len(manifest[k]) for k in ("train", "validation", "test", "normal_memory", "normal_calibration")},
              "effective_normal_calibration": len(calibration),
              "excluded_legacy_bank_overlap": [paths[i] for i in range(ntrain+nval+ntest, len(paths))
                                                if paths[i] in legacy_memory]}
    records = []
    for i, path in enumerate(paths):
        records.append({"path": path, "label": int(y[i]),
                        "partition": ("train" if i < ntrain else "validation" if i in validation
                                      else "test" if i in test else "excluded_bank_overlap"),
                        "scores": {mode: float(values[i]) for mode, values in score_sets.items()},
                        "baseline_pred": int(data["clip_scores"][i] >= config["threshold"]),
                        "selected_pred": int(score_sets[selected][i] >= thresholds[selected])})
    save_json(OUT / f"{product}_predictions.json", records)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    params = export_binary(classifier, "global")
    if selected in ("fast_fusion", "fusion"):
        params.update(export_binary(fast if selected == "fast_fusion" else fusion, "fusion"))
    if selected in ("multiscale", "fusion"):
        params["crop_bank"] = data["crop_bank"]
    if selected in ("patchcore", "fusion"):
        params["patchcore_bank"] = data["patchcore_bank"]
    np.savez_compressed(ARTIFACTS / f"{product}.npz", **params)
    threshold = thresholds[selected]
    normal_scores = score_sets[selected][calibration]
    high = max(threshold + max(abs(threshold) * .15, .01), float(np.quantile(normal_scores, .99)))
    if selected in ("global_binary", "fast_fusion", "fusion"):
        high = min(.9999, max(threshold + (1-threshold) * .5, float(np.quantile(normal_scores, .99))))
    metadata = {"product": product, "mode": selected, "name": NAMES[selected],
                "arrays": f"{product}.npz", "threshold": threshold, "t_high": high,
                "score_kind": "binary_model_score" if selected in ("global_binary", "fast_fusion", "fusion") else "distance",
                "selection_report": str((OUT / f"{product}_selection.json").relative_to(ROOT)),
                "normal_calibration_count": len(normal_scores)}
    save_json(ARTIFACTS / f"{product}.json", metadata)
    save_json(OUT / f"{product}_report.json", report)
    log(product, "SELECTED", selected, "validation", round(validation_results[selected]["accuracy"],4),
        "HELDOUT", round(test_results[selected]["accuracy"],4), "BASELINE", round(test_results["baseline"]["accuracy"],4))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", nargs="+", default=["metal_nut", "screw", "grid"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    products = json.loads((ROOT / "results/perception/banks/products.json").read_text(encoding="utf-8"))
    configs = [p for p in products if p["key"] in args.products]
    # Re-running an experiment after deployment must still compare with the
    # original anomaly threshold, never a probability threshold from the upgrade.
    configs = [{**p, "threshold": p.get("baseline_perception", {}).get("threshold", p["threshold"])}
               for p in configs]
    # Persist every split before feature extraction or model selection.
    manifests = {p["key"]: make_manifest(p["key"]) for p in configs}
    log("Loading fixed pretrained feature extractors; CPU threads=4")
    clip, patchcore = WinCLIPPatch("ViT-B-16", device="cpu"), PatchCoreFeatures()
    reports = {}
    with threadpool_limits(limits=4):
        for config in configs:
            key = config["key"]
            data = extract(key, config, clip, patchcore, manifests[key], args.batch_size)
            reports[key] = compare(key, config, data, manifests[key])
    save_json(OUT / "summary.json", {"created": datetime.now().isoformat(), "reports": reports,
              "selection": "validation only; no manual-review abstentions counted as correct",
              "patchcore_variant": "WideResNet50-2 V2 layer2+3, 3x3 pooling, image-stratified patch sampling, approximate 1024-center coreset, unweighted Euclidean nearest neighbor"})
    log("All requested comparisons complete:", OUT / "summary.json")


if __name__ == "__main__":
    main()
