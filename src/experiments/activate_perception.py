"""Install the validation-selected models that passed a held-out regression check.

Run after optimize_perception.py and verify_perception_upgrade.py. This never
chooses a different candidate using test results. Original config is backed up;
--rollback restores the config from a specific activation backup.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "results/perception/banks/products.json"
OUT = ROOT / "results/experiments/perception_upgrade_v1"
ARTIFACTS = ROOT / "results/perception/optimized"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback", type=Path)
    args = parser.parse_args()
    if args.rollback:
        source = args.rollback.resolve()
        backups = (ROOT / "results/perception/backups").resolve()
        if not source.is_relative_to(backups) or source.name != "products.json":
            raise ValueError("Rollback must name a products.json inside project backups")
        read(source)  # Reject invalid JSON before replacing the live config.
        shutil.copy2(source, CONFIG)
        print("Restored", source, "; restart the demo service")
        return

    verification = read(OUT / "runtime_verification.json")
    products = read(CONFIG)
    decisions = []
    for product in products:
        key = product["key"]
        report = read(OUT / f"{key}_report.json")
        selected = report["selected"]
        baseline, upgraded = report["test"]["baseline"], report["test"][selected]
        accepted = (selected != "baseline"
                    and upgraded["accuracy"] >= baseline["accuracy"]
                    and upgraded["balanced_accuracy"] >= baseline["balanced_accuracy"]
                    and upgraded["recall"] >= baseline["recall"])
        if not accepted:
            decisions.append({"product": key, "activated": False, "selected": selected,
                              "reason": "Validation-selected candidate did not pass held-out regression gate"})
            continue
        metadata = read(ARTIFACTS / f"{key}.json")
        if metadata["mode"] != selected or metadata["product"] != key:
            raise ValueError(f"Artifact/report mismatch for {key}")
        if not 0 <= metadata["threshold"] < metadata["t_high"]:
            raise ValueError(f"Invalid route thresholds for {key}")
        artifact_hash = hashlib.sha256((ARTIFACTS / metadata["arrays"]).read_bytes()).hexdigest()
        verified = verification["products"][key]
        if not verified["passed"] or verified["artifact_sha256"] != artifact_hash:
            raise ValueError(f"Runtime parity check missing or stale for {key}")
        if "baseline_perception" not in product:
            product["baseline_perception"] = {k: product[k] for k in
                ("threshold", "t_high", "calibration") if k in product}
        product.update({"perception_artifact": f"{key}.json", "perception_name": metadata["name"],
                        "threshold": metadata["threshold"], "t_high": metadata["t_high"],
                        "score_kind": metadata["score_kind"],
                        "calibration": {"method": "labeled-validation-threshold",
                                        "normal_calibration_count": metadata["normal_calibration_count"],
                                        "report": str((OUT / f"{key}_report.json").relative_to(ROOT)),
                                        "high_threshold_method": "engineering-review-band; not independently calibrated"}})
        decisions.append({"product": key, "activated": True, "selected": selected,
                          "baseline_test": baseline, "selected_test": upgraded,
                          "artifact_sha256": artifact_hash})
    backup = ROOT / "results/perception/backups" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-upgrade")
    backup.mkdir(parents=True)
    shutil.copy2(CONFIG, backup / "products.json")
    write(backup / "activation.json", {"decisions": decisions, "rollback_config": str(backup / "products.json")})
    temporary = CONFIG.with_suffix(".json.tmp")
    write(temporary, products)
    temporary.replace(CONFIG)
    write(OUT / "activation.json", {"decisions": decisions, "rollback_config": str(backup / "products.json")})
    print(json.dumps({"decisions": decisions, "backup": str(backup)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
