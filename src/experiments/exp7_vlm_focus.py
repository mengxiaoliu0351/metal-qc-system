"""Paired retrospective gray-zone pilot. Never writes inspection or training labels.

prepare freezes all three gray-zone normal images and six seeded defect images
per product. run compares single-image legacy review and full+focus review with
the same model, stored references, routing thresholds and automatic-action gates.
This enriched development set cannot establish independent production accuracy.
"""
import argparse
import collections
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def pixel_hash(image):
    return hashlib.sha256(str(image.size).encode() + image.convert("RGB").tobytes()).hexdigest()


def prepare(args):
    from PIL import Image
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT / "src"))
    from web.server import get_engine
    from cognition.visual_input import focus_views
    with args.source.open(encoding="utf-8-sig", newline="") as stream:
        gray = [r for r in csv.DictReader(stream) if r["route"] == "vlm_review"]
    rng = random.Random(20261105)
    selected = [r for r in gray if r["true_label"] == "good"]
    for product in sorted({r["product"] for r in gray}):
        pool = sorted((r for r in gray if r["product"] == product and r["true_label"] != "good"),
                      key=lambda r: r["session_id"])
        selected.extend(rng.sample(pool, min(6, len(pool))))
    samples, seen = [], set()
    for row in sorted(selected, key=lambda r: (r["product"], r["session_id"])):
        image = Image.open(ROOT / row["path"]).convert("RGB")
        fingerprint = pixel_hash(image)
        if fingerprint in seen:
            raise ValueError("Duplicate pixels in frozen pilot")
        seen.add(fingerprint)
        engine, _ = get_engine(row["product"])
        result = engine.detect(image)
        crops, geometry = focus_views(image, result["anomaly_map"], engine.preprocess)
        folder = args.output / row["session_id"]
        folder.mkdir(parents=True, exist_ok=True)
        image.save(folder / "full.png")
        for index, crop in enumerate(crops, 1):
            crop.save(folder / f"focus{index}.png")
        samples.append({"session_id": row["session_id"], "product": row["product"],
                        "truth": "normal" if row["true_label"] == "good" else "defect",
                        "pixel_hash": fingerprint, "focus": geometry, "source_path": row["path"]})
        print(json.dumps({"prepared": len(samples), "total": len(selected)}), flush=True)
    write_json(args.output / "manifest.json", {
        "selection": "all gray normal images + 6 defects per product; seed 20261105",
        "scope": "retrospective diagnostic pilot; independent blind-test status: false",
        "samples": samples})


def load_candidate(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summaries = {}
    rows = []
    arms = ["legacy", "focus"]
    if list(output.glob("*_focus_candidate.json")):
        arms.append("focus_candidate")
    for arm in arms:
        selected = []
        for sample in manifest["samples"]:
            path = output / (sample["session_id"] + "_" + arm + ".json")
            if path.is_file():
                selected.append(json.loads(path.read_text(encoding="utf-8")))
        rows.extend(selected)
        summaries[arm] = {
            "completed": len(selected),
            "api_success": sum(bool(r["cognition"].get("ok")) for r in selected),
            "conclusions": dict(collections.Counter(r["cognition"].get("conclusion", "failed") for r in selected)),
            "proposed_actions_before_audit": dict(collections.Counter(r["action"] for r in selected)),
            "defects_auto_normal": sum(r["truth"] == "defect" and r["action"] == "normal" for r in selected),
            "normals_auto_isolated": sum(r["truth"] == "normal" and r["action"] == "defect" for r in selected),
            "visual_defects_as_normal": sum(r["truth"] == "defect" and r["cognition"].get("conclusion") in ("合格", "正常") for r in selected),
            "visual_normals_as_defect": sum(r["truth"] == "normal" and r["cognition"].get("conclusion") == "不合格" for r in selected),
            "valid_nonempty_citations": sum(bool(r["cognition"].get("citation_check", {}).get("ok")
                                                 and r["cognition"].get("citation_check", {}).get("has_citations")) for r in selected),
            "median_seconds": round(statistics.median([r["seconds"] for r in selected]), 3) if selected else None,
            "usage_captured": sum(bool(r["cognition"].get("usage")) for r in selected),
            "input_tokens": sum(r["cognition"].get("usage", {}).get("prompt_tokens", 0) for r in selected)
                            if any(r["cognition"].get("usage") for r in selected) else None,
            "output_tokens": sum(r["cognition"].get("usage", {}).get("completion_tokens", 0) for r in selected)
                             if any(r["cognition"].get("usage") for r in selected) else None,
        }
    summary = {"scope": manifest["scope"], "samples": len(manifest["samples"]),
               "normal": sum(s["truth"] == "normal" for s in manifest["samples"]),
               "defect": sum(s["truth"] == "defect" for s in manifest["samples"]), "arms": summaries}
    write_json(output / "summary.json", summary)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["session_id", "product", "truth", "arm", "action", "seconds", "conclusion", "defect_type", "evidence_reason"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in ("session_id", "product", "truth", "arm", "action", "seconds")}
                            | {"conclusion": row["cognition"].get("conclusion"), "defect_type": row["cognition"].get("defect_type"),
                               "evidence_reason": row["evidence"]["reason"]})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def run(args):
    from PIL import Image
    sys.path.insert(0, "/app/src")
    from cognition import vlm_api as baseline
    from review.disposition import validate_references
    candidate = load_candidate("candidate_vlm", args.candidate / "cognition/vlm_api.py")
    comparison = None
    if args.candidate_model:
        comparison = load_candidate("comparison_vlm", args.candidate / "cognition/vlm_api.py")
        comparison.DASHSCOPE_MODEL = args.candidate_model
    policy = load_candidate("candidate_disposition", args.candidate / "review/disposition.py")
    manifest = json.loads((args.fixtures / "manifest.json").read_text(encoding="utf-8"))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Refuse to change a frozen experiment manifest")
    write_json(manifest_path, manifest)
    history = Path("/app/results/history/2026-11-04")
    configs = {p["key"]: p for p in json.loads(Path("/app/results/perception/banks/products.json").read_text())}
    jobs = []
    for sample in manifest["samples"]:
        sid = sample["session_id"]
        record_path = next(history.glob("*_" + sid + ".json"))
        record = json.loads(record_path.read_text(encoding="utf-8"))
        original = Image.open(record["input_png"]).convert("RGB")
        if pixel_hash(original) != sample["pixel_hash"]:
            raise ValueError("Source image changed: " + sid)
        profile = json.loads(Path("/app/results/automation_profiles", sample["product"] + ".json").read_text())
        arms = ["legacy", "focus"] + (["focus_candidate"] if comparison else [])
        for arm in arms:
            path = output / (sid + "_" + arm + ".json")
            if not path.exists():
                jobs.append((sample, record, profile, arm, path))

    def evaluate(job):
        sample, record, profile, arm, path = job
        refs = record.get("references", [])
        texts = [f"[{r['id']}] {r['standard']} {r['section']}: {r['text']}" for r in refs]
        gate = record["gate"]
        product = sample["product"]
        folder = args.fixtures / sample["session_id"]
        started = time.perf_counter()
        if arm == "legacy":
            hint_type = (record.get("supervised_class") or {}).get("label")
            if not hint_type or hint_type == "正常":
                probabilities = record.get("classification") or {}
                hint_type = max(probabilities, key=probabilities.get) if probabilities else "表面缺陷"
            hint = f"异常分数{record['score']:.4f}（介于{gate['threshold']}与{gate['t_high']}之间），疑似{hint_type}缺陷"
            cognition = baseline.analyze(str(folder / "full.png"), baseline.build_prompt(hint, texts))
        else:
            crops = [Image.open(p).convert("RGB") for p in sorted(folder.glob("focus*.png"))]
            hint = f"当前产品：{configs[product]['name']}。请独立观察整图及局部；本地模型将本图分入待分析区。"
            api = comparison if arm == "focus_candidate" else candidate
            cognition = api.analyze(str(folder / "full.png"), api.build_prompt(hint, texts),
                                          focus_images=crops, visual_review=True)
            cognition["visual_input"] = sample["focus"]
        seconds = round(time.perf_counter() - started, 3)
        cognition = validate_references(cognition, refs)
        evidence = policy.gray_decision_evidence(cognition, record["score"], gate["threshold"], gate["t_high"])
        action = evidence["proposal"]
        disposition = record["disposition"]
        if (not record.get("quality_check", {}).get("ok", False)
                or not disposition.get("scope_check", {}).get("ok", False)
                or profile["model_key"] != disposition.get("model_key")):
            action = "abstain"
        if action == "normal" and not profile["allow_normal"]:
            action = "abstain"
        if action == "defect" and not profile["allow_isolate"]:
            action = "abstain"
        result = {"session_id": sample["session_id"], "product": product, "truth": sample["truth"],
                  "arm": arm, "seconds": seconds, "action": action, "evidence": evidence, "cognition": cognition,
                  "model_requested": comparison.DASHSCOPE_MODEL if arm == "focus_candidate" else candidate.DASHSCOPE_MODEL,
                  "source_score": record["score"],
                  "source_gate": gate, "references": refs, "source_disposition": disposition,
                  "profile": {k: profile[k] for k in ("model_key", "allow_normal", "allow_isolate")}}
        write_json(path, result)
        return {"session_id": sample["session_id"], "arm": arm, "action": action, "seconds": seconds}

    print(json.dumps({"pending_calls": len(jobs), "model": candidate.DASHSCOPE_MODEL}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for index, result in enumerate(executor.map(evaluate, jobs), 1):
            print(json.dumps({"completed": index, "total": len(jobs), **result}), flush=True)
    summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("run")
    evaluate.add_argument("--fixtures", type=Path, required=True)
    evaluate.add_argument("--candidate", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--candidate-model", help="Optional separate model instance; never changes the live model")
    summary = sub.add_parser("summarize")
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "run":
        run(args)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
