"""Compare exported online inference against frozen offline predictions.

Uses real images/weights, never VLM. Times warm inference separately from loading.
Known development images are regression cases, not accuracy evidence.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from perception.winclip_patch import WinCLIPPatch
from perception.feature_backends import OptimizedPerception, PatchCoreFeatures

OUT = ROOT / "results/experiments/perception_upgrade_v1"
ARTIFACTS = ROOT / "results/perception/optimized"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    detector = WinCLIPPatch("ViT-B-16", device="cpu")
    pc = None

    def patchcore():
        nonlocal pc
        if pc is None:
            pc = PatchCoreFeatures()
        return pc

    products = read(ROOT / "results/perception/banks/products.json")
    verification = {"products": {}}
    with threadpool_limits(limits=4):
        for config in products:
            key = config["key"]
            metadata = read(ARTIFACTS / f"{key}.json")
            if metadata["mode"] == "baseline":
                continue
            records = read(OUT / f"{key}_predictions.json")
            cases = [next(r for r in records if r["partition"] == "test" and r["label"] == label)
                     for label in (0, 1)]
            if key == "metal_nut":
                cases += [next(r for r in records if Path(r["path"]).as_posix().endswith(suffix))
                          for suffix in ("scratch/001.png", "bent/000.png")]
            detector.bank = np.load(ROOT / "results/perception/banks" / config["bank"])
            optimized = OptimizedPerception(ARTIFACTS / f"{key}.json", patchcore_factory=patchcore)
            results = []
            for record in cases:
                with Image.open(ROOT / record["path"]) as source:
                    image = source.convert("RGB")
                started = time.perf_counter()
                baseline = detector.detect(image)
                actual = optimized.apply(detector, image, baseline)
                elapsed = time.perf_counter() - started
                expected = record["scores"][metadata["mode"]]
                difference = abs(actual["score"] - expected)
                if difference > 1e-4:
                    raise AssertionError(f"Offline/online mismatch: {key} {record['path']} {difference}")
                prediction = int(actual["score"] >= metadata["threshold"])
                if prediction != record["selected_pred"]:
                    raise AssertionError("Exported threshold/prediction mismatch")
                results.append({"path": record["path"], "partition": record["partition"],
                                "label": record["label"], "prediction": prediction,
                                "score": actual["score"], "baseline_score": baseline["score"],
                                "expected_score": expected, "difference": difference,
                                "seconds": elapsed})
                print(key, Path(record["path"]).parent.name, Path(record["path"]).name,
                      "score", round(actual["score"], 6), "seconds", round(elapsed, 3), flush=True)
            verification["products"][key] = {
                "passed": True, "cases": results,
                "artifact_sha256": hashlib.sha256((ARTIFACTS / metadata["arrays"]).read_bytes()).hexdigest()}
    (OUT / "runtime_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print("All exported inference checks passed", flush=True)


if __name__ == "__main__":
    main()
