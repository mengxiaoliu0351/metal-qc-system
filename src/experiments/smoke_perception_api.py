"""Smoke-test the running local demo using verified low/high-risk cases only.

No selected case enters the VLM route. Real detections are saved in the app's
history with a recognizable validation filename prefix.
"""
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/experiments/perception_upgrade_v1"


def main():
    session = requests.Session()
    session.trust_env = False
    base = "http://127.0.0.1:8000"
    response = session.get(base + "/api/products", timeout=10)
    response.raise_for_status()
    products = {p["key"]: p for p in response.json()["products"]}
    verification = json.loads((OUT / "runtime_verification.json").read_text(encoding="utf-8"))
    results = []
    for key, checked in verification["products"].items():
        config = products[key]
        for case in checked["cases"]:
            score = case["score"]
            if config["threshold"] <= score <= config["t_high"]:
                continue  # Explicitly avoid any cloud call during smoke tests.
            source = ROOT / case["path"]
            with source.open("rb") as stream:
                response = session.post(base + "/api/detect", data={"product": key},
                    files={"file": (f"validation_{key}_{source.parent.name}_{source.name}", stream, "image/png")},
                    timeout=120)
            response.raise_for_status()
            data = response.json()
            if not data.get("ok"):
                raise AssertionError(data)
            expected_route = "low_risk" if score < config["threshold"] else "high_risk"
            if data["route"] != expected_route or abs(data["score"] - score) > 1e-4:
                raise AssertionError("Running API differs from verified artifact")
            if data["perception_prediction"] != ("defect" if case["prediction"] else "normal"):
                raise AssertionError("API prediction mismatch")
            if not data.get("supervised_class") or data["perception_method"] != config["method"]:
                raise AssertionError("Classifier/model metadata missing")
            result = {"product": key, "path": case["path"], "score": data["score"],
                      "threshold": data["threshold"], "t_high": data["t_high"],
                      "route": data["route"], "method": data["perception_method"],
                      "prediction": data["perception_prediction"],
                      "type_prediction": data["supervised_class"]["label"],
                      "conclusion": data["cognition"]["conclusion"],
                      "session_id": data["session_id"], "seconds": data["time_perception"]}
            results.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)
    for path in ("/", "/static/app.js", "/static/style.css"):
        response = session.get(base + path, timeout=10)
        response.raise_for_status()
    (OUT / "api_smoke.json").write_text(json.dumps({"passed": True, "vlm_calls": 0, "cases": results},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
