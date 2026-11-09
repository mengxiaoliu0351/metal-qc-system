"""Exercise iteration with dataset references; never impersonates a human inspector."""
import argparse
import json
from pathlib import Path
import time

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve-source-labels", action="store_true", required=True)
    parser.add_argument("--mode", choices=("model", "threshold"), default="model")
    args = parser.parse_args()
    session = requests.Session()
    session.trust_env = False
    def request(method, path, **kwargs):
        response = session.request(method, args.endpoint.rstrip("/") + path, timeout=90, **kwargs)
        response.raise_for_status()
        return response.json()
    batch = request("GET", "/api/batches/" + args.batch)
    if batch["purpose"] != "simulation" or batch["status"] != "completed" or batch["counts"]["failed"]:
        raise ValueError("Use only a completed simulation batch with all source labels available")
    approvals = []
    for item in batch["items"]:
        truth = item.get("dataset_truth")
        if truth not in ("normal", "defect"): raise ValueError("Missing dataset reference")
        response = request("POST", "/api/review/confirm", json={
            "session_id": item["result"]["session_id"], "verdict": "ok" if truth == "normal" else "defect",
            "reviewer": "DATASET_LABEL_REPLAY", "use_for_iteration": True,
            "note": "Engineering acceptance: publisher-label replay; not a real inspector review."})
        if response["annotation"]["verification_method"] != "dataset_label_replay":
            raise ValueError("Reference replay was not marked separately from operator review")
        approvals.append(item["id"])
    run = request("POST", "/api/iterations", json={"product": batch["product"], "mode": args.mode})
    result = {"batch": batch["id"], "product": batch["product"], "approved": approvals,
              "verification_method": "dataset_label_replay", "iteration": run}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"approved": len(approvals), "iteration": run}, ensure_ascii=True), flush=True)
    while True:
        state = next(r for r in request("GET", "/api/iterations")["runs"] if r["id"] == run["id"])
        if state["status"] not in ("queued", "running"):
            result["state"] = state
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
            print(json.dumps({k: state.get(k) for k in ("id", "status", "message")}, ensure_ascii=True))
            break
        time.sleep(5)


if __name__ == "__main__": main()
