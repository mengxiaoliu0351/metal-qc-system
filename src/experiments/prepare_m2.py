"""Freeze the publisher's split, audit pixel duplicates, then import via website APIs."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import threading

import numpy as np
from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[2]


def audit(source, output):
    source = source.resolve()
    entries, groups = [], defaultdict(list)
    for path in sorted(source.rglob("*")):
        if path.suffix.lower() not in (".bmp", ".png", ".jpg", ".jpeg"):
            continue
        relative = path.relative_to(source)
        if len(relative.parts) != 3 or relative.parts[0] not in ("Train", "Test") or relative.parts[1] not in ("OK", "NG"):
            raise ValueError("Unexpected source folders")
        with Image.open(path) as original:
            image = original.convert("RGB")
            fingerprint = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            gray = np.asarray(image.convert("L").resize((9, 8)))
            bits = (gray[:, 1:] > gray[:, :-1]).ravel()
            dhash = sum(int(bit) << i for i, bit in enumerate(bits))
        item = {"source_path": str(path.relative_to(ROOT)), "source_partition": relative.parts[0],
                "label": "normal" if relative.parts[1] == "OK" else "defect", "pixel_hash": fingerprint,
                "dhash": f"{dhash:016x}", "role": None, "excluded": None}
        entries.append(item); groups[fingerprint].append(item)
    for items in groups.values():
        if len({s["label"] for s in items}) > 1:
            for item in items: item["excluded"] = "conflicting_source_labels"
            continue
        items.sort(key=lambda s: (s["source_partition"] != "Test", s["source_path"]))
        for item in items[1:]: item["excluded"] = "duplicate_pixels"
    rng = random.Random(20261105)
    for label in ("normal", "defect"):
        pool = [s for s in entries if not s["excluded"] and s["source_partition"] == "Train" and s["label"] == label]
        rng.shuffle(pool)
        # Four disjoint development roles. Test remains exactly the retained official Test split.
        bank_n = min(80, len(pool) // 4) if label == "normal" else 0
        for item in pool[:bank_n]: item["role"] = "bank"
        remaining = pool[bank_n:]
        cal_n = min(100, len(remaining) // 3)
        qual_n = min(100, len(remaining) // 3)
        for item in remaining[:cal_n]: item["role"] = "calibration"
        for item in remaining[cal_n:cal_n+qual_n]: item["role"] = "qualification"
        training = remaining[cal_n+qual_n:]
        for item in training[:200]: item["role"] = "train"
        for item in training[200:]: item["role"] = "feedback"
    for item in entries:
        if not item["excluded"] and item["source_partition"] == "Test": item["role"] = "test"
    counts = Counter((s["role"], s["label"]) for s in entries if not s["excluded"])
    summary = {"source_images": len(entries), "retained": sum(not s["excluded"] for s in entries),
               "excluded": dict(Counter(s["excluded"] for s in entries if s["excluded"])),
               "roles": {r: {label: counts[r, label] for label in ("normal", "defect")}
                         for r in ("bank", "train", "calibration", "qualification", "test", "feedback")}}
    manifest = {"dataset": "M2 screw defect detection dataset V2, HEAD", "doi": "10.17632/2y6mcsyrpy.2",
                "source": "https://data.mendeley.com/datasets/2y6mcsyrpy/2", "license": "CC BY 4.0",
                "physical_identity": "not provided; image-level split only", "seed": 20261105,
                "summary": summary, "images": entries}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def upload(manifest_path, endpoint, state_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    session = requests.Session()
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf8"))
    else:
        response = session.post(endpoint + "/api/product-catalog", json={"name": "M2 金属螺丝",
            "sku": "M2（公开数据规格）", "view": "头部俯视", "source": manifest["source"], "license": manifest["license"]}, timeout=30)
        response.raise_for_status()
        state = {"key": response.json()["key"], "imported": []}
        state_path.write_text(json.dumps(state), encoding="utf8")
    imported = set(state["imported"])
    samples = [s for s in manifest["images"] if not s["excluded"]]
    pending = [s for s in samples if s["pixel_hash"] not in imported]
    local = threading.local()
    def send(sample):
        if not hasattr(local, "session"):
            local.session = requests.Session()
        with Image.open(ROOT / sample["source_path"]) as image:
            stream = io.BytesIO(); image.convert("RGB").save(stream, format="PNG")
        response = local.session.post(endpoint + f"/api/product-catalog/{state['key']}/samples",
            files={"file": (Path(sample["source_path"]).name, stream.getvalue(), "image/png")},
            data={"role": sample["role"], "label": sample["label"]}, timeout=60)
        response.raise_for_status()
        return sample["pixel_hash"]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for fingerprint in executor.map(send, pending):
            imported.add(fingerprint); state["imported"] = sorted(imported)
            temp = state_path.with_suffix(".tmp"); temp.write_text(json.dumps(state), encoding="utf8"); temp.replace(state_path)
            if len(imported) % 50 == 0: print(json.dumps({"imported": len(imported), "total": len(samples)}), flush=True)
    print(json.dumps({"key": state["key"], "imported": len(imported)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("audit")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    imp = sub.add_parser("import")
    imp.add_argument("--manifest", type=Path, required=True)
    imp.add_argument("--endpoint", default="http://127.0.0.1:8000")
    imp.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "audit": audit(args.source, args.output)
    else: upload(args.manifest, args.endpoint, args.state)


if __name__ == "__main__": main()
