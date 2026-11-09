"""Retrospective cross-product bank/threshold test for the original CLIP patch score.

All three banks are rebuilt from the same frozen CLIP weights. Each product
uses the first 80 sorted train/good images for its bank, the next 40 for
threshold calibration, and test/* for evaluation. No website artifacts change.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from perception.winclip_patch import WinCLIPPatch

OUT = ROOT / "results" / "experiments" / "cross_product_bank_20261104"
JOURNAL = OUT / "scores.jsonl"
BANK_DIR = OUT / "banks"
PRODUCTS = ("screw", "grid")
WEIGHTS = ROOT / "weights" / "CLIP-ViT-B-16-open_clip_model.safetensors"
OLD_THRESHOLD = 0.2345


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_hash(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def split_paths(product):
    base = ROOT / "data" / "mvtec" / product
    normal = sorted((base / "train" / "good").glob("*.png"))
    bank, calibration = normal[:80], normal[80:120]
    test = sorted((base / "test").glob("*/*.png"))
    if len(bank) != 80 or len(calibration) != 40 or not test:
        raise ValueError(f"Incomplete MVTec category: {product}")
    return bank, calibration, test


def verify_split(product, bank, calibration, test):
    groups = {"bank": bank, "calibration": calibration, "test": test}
    seen = {}
    for group, paths in groups.items():
        for path in paths:
            h = pixel_hash(path)
            if h in seen:
                raise ValueError(f"Pixel-identical images across partitions: {seen[h]}, {path}")
            seen[h] = path
    return {group: [p.relative_to(ROOT).as_posix() for p in paths] for group, paths in groups.items()}


def make_bank(detector, images, product):
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    path = BANK_DIR / f"{product}.npy"
    if path.exists():
        bank = np.load(path)
        if bank.shape != (80 * 196, 768):
            raise ValueError(f"Unexpected cached bank shape: {path}")
        print("loaded experiment bank", product, flush=True)
        return bank, path
    patches = []
    with threadpool_limits(limits=4):
        for start in range(0, len(images), 8):
            batch = []
            for image_path in images[start:start + 8]:
                with Image.open(image_path) as image:
                    batch.append(image.convert("RGB"))
            features, _ = detector.extract_batch(batch)
            patches.append(features.reshape(-1, features.shape[-1]))
            print("building bank", product, start + len(batch), "/", len(images), flush=True)
    bank = np.concatenate(patches).astype(np.float32, copy=False)
    np.save(path, bank)
    return bank, path


def calibrated_threshold(scores):
    values = np.asarray(scores, dtype=np.float64)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return round(center + 1.5 * mad, 4), center, mad


def wilson(success, total):
    z = 1.959963984540054
    p = success / total
    denom = 1 + z * z / total
    midpoint = (p + z * z / (2 * total)) / denom
    radius = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** .5) / denom
    return [round(midpoint - radius, 4), round(midpoint + radius, 4)]


def metrics(rows, score_key, threshold):
    truth = np.array([row["label"] != "good" for row in rows], dtype=bool)
    scores = np.array([row[score_key] for row in rows], dtype=float)
    pred = scores >= threshold
    tn = int((~truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    tp = int((truth & pred).sum())
    return {"threshold": threshold, "n": len(rows), "normal_n": tn + fp, "defect_n": fn + tp,
            "confusion_tn_fp_fn_tp": [tn, fp, fn, tp],
            "false_positive_rate": fp / (tn + fp), "false_positive_rate_95ci": wilson(fp, tn + fp),
            "defect_recall": tp / (tp + fn), "defect_recall_95ci": wilson(tp, tp + fn),
            "balanced_accuracy": .5 * (tn / (tn + fp) + tp / (tp + fn)),
            "auroc": float(roc_auc_score(truth, scores)), "score_mean_normal": float(scores[~truth].mean()),
            "score_mean_defect": float(scores[truth].mean())}


def read_journal():
    if not JOURNAL.exists():
        return {}
    rows = [json.loads(line) for line in JOURNAL.read_text(encoding="utf-8").splitlines() if line]
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate score rows in journal")
    return {row["path"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many new images for a pilot")
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    splits = {}
    for product in ("metal_nut",) + PRODUCTS:
        bank, calibration, test = split_paths(product)
        splits[product] = verify_split(product, bank, calibration, test)
        print(product, "bank", len(bank), "calibration", len(calibration), "test", len(test), flush=True)
    detector = WinCLIPPatch("ViT-B-16", device="cpu")
    banks = {}
    bank_paths = {}
    for product in ("metal_nut",) + PRODUCTS:
        banks[product], bank_paths[product] = make_bank(detector, [ROOT / p for p in splits[product]["bank"]], product)
    source = banks["metal_nut"]
    manifest = {"protocol": "one frozen CLIP model; first 80 train/good for bank, next 40 train/good for calibration, all test/* for target evaluation",
                "historical_threshold_auxiliary": OLD_THRESHOLD,
                "threshold_rule": "median + 1.5 * MAD on 40 product normal calibration images; rounded to 4 decimals",
                "sha256": {"weights": digest_file(WEIGHTS),
                           **{product + "_bank": digest_file(path) for product, path in bank_paths.items()}},
                "splits": splits}
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise ValueError("Frozen manifest or model/bank hash changed; refusing to mix runs")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    done = read_journal()
    planned = [(product, group, ROOT / rel)
               for product, split in splits.items()
               for group in ("calibration", "test") for rel in split[group]]
    if any(row["path"] not in {p.relative_to(ROOT).as_posix() for _, _, p in planned} for row in done.values()):
        raise ValueError("Journal includes paths outside frozen split")
    print("planned", len(planned), "completed", len(done), flush=True)
    processed = 0
    with threadpool_limits(limits=4):
        for i, (product, group, path) in enumerate(planned, 1):
            rel = path.relative_to(ROOT).as_posix()
            if rel in done:
                continue
            started = time.perf_counter()
            with Image.open(path) as image:
                patches, _ = detector.extract_features(image.convert("RGB"))
            source_score = float(np.maximum(0, 1 - (patches @ source.T).max(axis=1)).max())
            target_score = (float(np.maximum(0, 1 - (patches @ banks[product].T).max(axis=1)).max())
                            if product != "metal_nut" else None)
            row = {"path": rel, "product": product, "group": group, "label": path.parent.name,
                   "source_score": source_score, "target_score": target_score,
                   "elapsed_seconds": round(time.perf_counter() - started, 3)}
            with JOURNAL.open("a", encoding="utf-8") as output:
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            done[rel] = row
            processed += 1
            if processed <= 3 or processed % 10 == 0:
                print(f"{i}/{len(planned)} {product} {group} {processed} new; {row['elapsed_seconds']}s", flush=True)
            if args.limit and processed >= args.limit:
                break
    if len(done) != len(planned):
        print("incomplete", len(done), "/", len(planned), "resume by rerunning", flush=True)
        return
    results = {}
    detail = []
    source_cal_rows = [done[p] for p in splits["metal_nut"]["calibration"]]
    source_threshold, source_median, source_mad = calibrated_threshold([r["source_score"] for r in source_cal_rows])
    source_test_rows = [done[p] for p in splits["metal_nut"]["test"]]
    source_control = metrics(source_test_rows, "source_score", source_threshold)
    for product in PRODUCTS:
        split = splits[product]
        cal = [done[p] for p in split["calibration"]]
        test = [done[p] for p in split["test"]]
        source_on_target_cal, source_on_target_median, source_on_target_mad = calibrated_threshold([r["source_score"] for r in cal])
        target_cal, target_median, target_mad = calibrated_threshold([r["target_score"] for r in cal])
        arms = {
            "A_direct_transfer": metrics(test, "source_score", source_threshold),
            "B_threshold_only": metrics(test, "source_score", source_on_target_cal),
            "C_bank_only": metrics(test, "target_score", source_threshold),
            "D_target_bank_and_threshold": metrics(test, "target_score", target_cal),
            "A0_historical_threshold_auxiliary": metrics(test, "source_score", OLD_THRESHOLD),
        }
        results[product] = {"calibration": {"source_domain_threshold": source_threshold, "source_domain_median": source_median,
                                               "source_domain_mad": source_mad,
                                               "source_bank_target_normal_threshold": source_on_target_cal,
                                               "source_bank_target_normal_median": source_on_target_median,
                                               "source_bank_target_normal_mad": source_on_target_mad,
                                               "target_bank_threshold": target_cal,
                                               "target_median": target_median, "target_mad": target_mad},
                            "arms": arms}
        for row in test:
            detail.append({**row,
                           "A_prediction": "defect" if row["source_score"] >= source_threshold else "normal",
                           "B_prediction": "defect" if row["source_score"] >= source_on_target_cal else "normal",
                           "C_prediction": "defect" if row["target_score"] >= source_threshold else "normal",
                           "D_prediction": "defect" if row["target_score"] >= target_cal else "normal",
                           "A0_prediction": "defect" if row["source_score"] >= OLD_THRESHOLD else "normal"})
    (OUT / "summary.json").write_text(json.dumps({"manifest": "manifest.json", "source_in_domain_control": source_control,
                                                  "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    print(json.dumps({"source_in_domain_control": source_control, "results": results}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
