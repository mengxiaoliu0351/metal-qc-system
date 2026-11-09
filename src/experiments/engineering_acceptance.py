"""Freeze a calibration-only baseline, then score a completed website batch."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from review.workflow import measurement, read, write


def freeze(args):
    configs = read(ROOT / "results/perception/banks/products.json")
    config = next(c for c in configs if c["key"] == args.product)
    folder = ROOT / "results" / config["review_benchmark"]
    split = read(folder / (args.product + "_split.json"))
    with np.load(folder / (args.product + "_features.npz"), allow_pickle=False) as data:
        index = {str(p): i for i, p in enumerate(data["paths"])}
        normal_cal = [index[p] for p in split["validation"] if Path(p).parent.name == "good"]
        threshold = float(np.quantile(data["clip_scores"][normal_cal], .99))
    baseline = {"product": args.product, "mode": "normal-memory-only", "threshold": threshold,
                "t_high": max(threshold * 1.5, float(np.nextafter(threshold, np.inf))),
                "calibration_normal_n": len(normal_cal), "method": "99th percentile of normal calibration; no defect labels",
                "current_config": config, "scope": "cold-start baseline and labelled enhancement use different label budgets"}
    path = args.output / "baseline.json"
    if path.exists() and read(path) != baseline:
        raise ValueError("Baseline already frozen; use a new experiment directory")
    write(path, baseline)
    print(json.dumps(baseline, ensure_ascii=True), flush=True)


def analyze(args):
    baseline = read(args.output / "baseline.json")
    response = requests.get(args.endpoint + "/api/batches/" + args.batch, timeout=90)
    response.raise_for_status()
    batch = response.json()
    if batch["status"] not in ("completed", "completed_with_errors"):
        raise ValueError("Batch has not completed")
    if batch["product"] != baseline["product"]:
        raise ValueError("Test product differs from frozen baseline")
    config = batch["recipe"]["config"]
    if config != baseline["current_config"]:
        raise ValueError("Product configuration changed after baseline was frozen")
    scored = [i for i in batch["items"] if i["status"] == "completed" and i.get("dataset_truth") in ("normal", "defect")
              and isinstance((i.get("result") or {}).get("score"), (int, float))]
    labels = np.array([i["dataset_truth"] == "defect" for i in scored], dtype=int)
    current = np.array([i["result"]["score"] for i in scored])
    raw = np.array([i["result"].get("baseline_score", i["result"]["score"]) for i in scored])
    times = sorted(i["result"]["time_total"] for i in scored if isinstance(i["result"].get("time_total"), (float, int)))
    actions = [(i["result"].get("disposition") or {}) for i in batch["items"] if i.get("result")]
    direct = [i for i in scored if (i["result"].get("disposition") or {}).get("auto_result") and not i["result"]["disposition"]["requires_review"]]
    review = sum(bool(a.get("requires_review")) for a in actions)
    calls = [i["result"] for i in batch["items"] if (i.get("result") or {}).get("vlm_called")]
    summary = {"product": batch["product"], "batch": batch["id"], "endpoint": args.endpoint,
        "total": batch["total"], "completed": batch["counts"]["completed"], "failed": batch["counts"]["failed"],
        "local_scored": len(scored), "normal": int(np.sum(labels == 0)), "defect": int(np.sum(labels == 1)),
        "baseline": measurement(labels, raw, baseline["threshold"], baseline["t_high"]),
        "current_local": measurement(labels, current, config["threshold"], config["t_high"]),
        "system": {"direct_automatic": len(direct), "review_tasks": review,
            "direct_automatic_rate": len(direct) / batch["total"] if batch["total"] else None,
            "direct_normal_misses": sum(i["dataset_truth"] == "defect" and i["result"]["disposition"]["auto_result"] == "normal" for i in direct),
            "direct_isolation_false_positives": sum(i["dataset_truth"] == "normal" and i["result"]["disposition"]["auto_result"] == "defect" for i in direct),
            "vlm_path_attempts": len(calls), "vlm_success": sum(bool((r.get("cognition") or {}).get("ok")) for r in calls),
            "median_service_seconds": statistics.median(times) if times else None,
            "p95_service_seconds": times[max(0, math.ceil(.95 * len(times)) - 1)] if times else None},
        "scope": "publisher labels and image-level retained test split; physical identities not supplied; pending review has no final correctness claim"}
    write(args.output / (args.batch + "_summary.json"), summary)
    write(args.output / (args.batch + "_batch.json"), batch)
    print(json.dumps(summary, ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--product", required=True)
    f.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--batch", required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--endpoint", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    freeze(args) if args.mode == "freeze" else analyze(args)


if __name__ == "__main__": main()
