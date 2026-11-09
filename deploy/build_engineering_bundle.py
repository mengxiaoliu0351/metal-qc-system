"""Package code and one new catalogue product without replacing live history/config."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def bundle(path, files):
    temporary = path.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        for file in sorted(files): archive.add(file, arcname=file.relative_to(ROOT).as_posix())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"product_[a-f0-9]{12}", args.product): raise ValueError("Invalid product key")
    args.output.mkdir(parents=True, exist_ok=True)
    config = next(p for p in json.loads((ROOT / "results/perception/banks/products.json").read_text(encoding="utf8")) if p["key"] == args.product)
    code = {p for p in (ROOT / "src").rglob("*") if p.is_file() and not any(x in p.parts for x in ("__pycache__", "weights", "experiments")) and p.suffix not in (".pyc", ".pyo")}
    code.update(ROOT / p for p in ("deploy/Dockerfile", "deploy/requirements-cloud.txt"))
    data = {p for p in (ROOT / "results/product_catalog" / args.product).rglob("*") if p.is_file() and not p.name.endswith(".tmp")}
    data.add(ROOT / "results/perception/banks" / config["bank"])
    metadata = ROOT / "results/perception/optimized" / config["perception_artifact"]
    data.add(metadata); data.add(metadata.parent / json.loads(metadata.read_text(encoding="utf8"))["arrays"])
    data.update(ROOT / p for p in ("data/external/m2_v2/source.json", "data/external/m2_v2/head-manifest.json"))
    for path in code | data:
        if path.suffix in (".pem", ".key") or path.name.startswith(".env"):
            raise ValueError("Credentials cannot enter a bundle")
    config_path = args.output / "product-config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf8")
    policy_path = ROOT / "results/inspection_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf8")) if policy_path.exists() else {"holds": {}}
    hold_path = args.output / "release-gate.json"
    holds = {k: v for k, v in policy.get("holds", {}).items() if k.startswith(args.product + ":")}
    hold_path.write_text(json.dumps({"product": args.product, "holds": holds}, ensure_ascii=False, indent=2), encoding="utf8")
    for label, files in (("code", code), ("product", data)):
        target = args.output / (label + ".tar.gz")
        if args.code_only and label == "product":
            if not target.exists(): raise ValueError("Product archive missing")
            continue
        bundle(target, files)
        print(json.dumps({"bundle": label, "bytes": target.stat().st_size}), flush=True)
    manifest = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in (args.output / "code.tar.gz", args.output / "product.tar.gz", config_path, hold_path)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print("Allowlisted engineering bundles ready", flush=True)


if __name__ == "__main__": main()
