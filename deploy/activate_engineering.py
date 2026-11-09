"""Verified app update with server-local backups and rollback. Run on the ECS host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.request


def run(arguments, capture=False):
    return subprocess.run(arguments, check=True, text=True, capture_output=capture)


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""): h.update(data)
    return h.hexdigest()


def safe_extract(archive_path, target, allowed):
    with tarfile.open(archive_path, "r:gz") as archive:
        for item in archive.getmembers():
            name = Path(item.name)
            if name.is_absolute() or ".." in name.parts or not (item.isfile() or item.isdir()) or not allowed(item.name):
                raise ValueError("Archive contains an unexpected entry")
        archive.extractall(target, filter="data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    root = Path("/opt/metalqc")
    if os.name != "posix" or not (root / "deploy/compose.yaml").is_file():
        raise ValueError("Run this command on the configured ECS host")
    args.stage, args.backup = args.stage.resolve(), args.backup.resolve()
    if not args.stage.is_relative_to(Path("/opt/metalqc-staging")) or not args.backup.is_relative_to(Path("/opt/metalqc-backups")):
        raise ValueError("Unexpected staging or backup location")
    if (args.backup / "source.tar.gz").exists():
        raise ValueError("Backup already exists; inspect prior deployment state before retrying")
    if shutil.disk_usage(root).free < 5 * 1024 ** 3:
        raise ValueError("At least 5GB free space is required for a recoverable update")
    manifest = json.loads((args.stage / "manifest.json").read_text())
    for name in ("code.tar.gz", "product.tar.gz", "product-config.json", "release-gate.json"):
        if checksum(args.stage / name) != manifest[name]["sha256"]:
            raise ValueError("Staged bundle checksum mismatch: " + name)
    product = json.loads((args.stage / "product-config.json").read_text())
    if not product["key"].startswith("product_"): raise ValueError("Unexpected product")
    release = json.loads((args.stage / "release-gate.json").read_text())
    if release["product"] != product["key"] or any(not k.startswith(product["key"] + ":") for k in release["holds"]):
        raise ValueError("Release evidence belongs to a different product")
    args.backup.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.backup.chmod(0o700)
    compose = ["docker", "compose", "-f", str(root / "deploy/compose.yaml")]
    container = run(compose + ["ps", "-q", "app"], capture=True).stdout.strip()
    image = run(["docker", "inspect", "--format", "{{.Image}}", container], capture=True).stdout.strip()
    pinned = "metalqc-backup:engineering-20261105"
    run(["docker", "tag", image, pinned])
    with tarfile.open(args.backup / "source.tar.gz", "w:gz", compresslevel=1) as archive:
        archive.add(root / "src", arcname="src", filter=lambda m: None if "__pycache__" in m.name else m)
        archive.add(root / "deploy/Dockerfile", arcname="deploy/Dockerfile")
        archive.add(root / "deploy/requirements-cloud.txt", arcname="deploy/requirements-cloud.txt")
    shutil.copy2(root / "runtime.env", args.backup / "runtime.env")
    (args.backup / "runtime.env").chmod(0o600)
    allowed_code = lambda n: n.startswith("src/") or n in ("src", "deploy", "deploy/Dockerfile", "deploy/requirements-cloud.txt")
    safe_extract(args.stage / "code.tar.gz", root, allowed_code)
    print("Source backed up; building candidate image while current service remains running", flush=True)
    try:
        run(compose + ["build", "app"])
    except Exception:
        safe_extract(args.backup / "source.tar.gz", root, allowed_code)
        raise
    allowed_data = lambda n: n.startswith("results/product_catalog/" + product["key"] + "/") or (
        n == "results/perception/banks/" + product["bank"]) or n in (
            "results/perception/optimized/" + product["perception_artifact"],
            "results/perception/optimized/" + product["perception_artifact"].replace(".json", ".npz"),
            "data/external/m2_v2/source.json", "data/external/m2_v2/head-manifest.json")
    safe_extract(args.stage / "product.tar.gz", root, allowed_data)
    for directory in (root / "results/product_catalog" / product["key"],):
        for path in [directory, *directory.rglob("*")]: os.chown(path, 10001, 10001)
    for path in (root / "results/perception/banks" / product["bank"],
                 root / "results/perception/optimized" / product["perception_artifact"],
                 root / "results/perception/optimized" / product["perception_artifact"].replace(".json", ".npz")):
        os.chown(path, 10001, 10001)
    registry_path = root / "results/perception/banks/products.json"
    policy_path = root / "results/inspection_policy.json"
    print("Candidate image ready; briefly stopping app to snapshot mutable state", flush=True)
    run(compose + ["stop", "app"])
    shutil.copy2(registry_path, args.backup / "products.json")
    had_policy = policy_path.is_file()
    if had_policy: shutil.copy2(policy_path, args.backup / "inspection_policy.json")
    try:
        # New product assets are already separately preserved in the staged bundle.
        def old_state_only(member):
            return None if product["key"] in member.name or member.name.endswith(".tmp") else member
        with tarfile.open(args.backup / "results.tar.gz", "w:gz", compresslevel=1) as archive:
            archive.add(root / "results", arcname="results", filter=old_state_only)
        registry = json.loads(registry_path.read_text())
        existing = next((p for p in registry if p["key"] == product["key"]), None)
        if existing and existing != product: raise ValueError("New product key is already bound to another version")
        if not existing:
            temporary = registry_path.with_suffix(".engineering.tmp")
            temporary.write_text(json.dumps(registry + [product], ensure_ascii=False, indent=2), encoding="utf8")
            os.chown(temporary, 10001, 10001)
            temporary.replace(registry_path)
        # Merge this version's failed-test veto; preserve all existing operational settings.
        if release["holds"]:
            policy = json.loads(policy_path.read_text()) if had_policy else {
                "version": 2, "mode": "assisted", "sample_rate": .10, "audit_period": "day", "holds": {}, "revision": "initial"}
            policy["holds"].update(release["holds"])
            policy["revision"] = "engineering-release-20261106"
            temporary = policy_path.with_suffix(".engineering.tmp")
            temporary.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf8")
            os.chown(temporary, 10001, 10001)
            temporary.replace(policy_path)
        run(compose + ["up", "-d", "--no-deps", "app"])
        for _ in range(30):
            try:
                with urllib.request.urlopen("http://127.0.0.1:18000/healthz", timeout=5) as response:
                    healthy = json.load(response).get("ok")
                if healthy: break
            except Exception:
                time.sleep(2)
        else: raise RuntimeError("New app failed health checks")
        with urllib.request.urlopen("http://127.0.0.1:18000/api/products", timeout=30) as response:
            online = json.load(response)["products"]
        if product["key"] not in {p["key"] for p in online}: raise RuntimeError("New product is missing")
        before = json.loads((args.backup / "products.json").read_text())
        current = json.loads(registry_path.read_text())
        if any(p not in current for p in before): raise RuntimeError("Original products changed")
        result = {"healthy": True, "product_count": len(online), "original_products_preserved": len(before),
                  "rollback_image": pinned, "image_before": image}
        (args.backup / "deployment.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    except Exception:
        if had_policy: shutil.copy2(args.backup / "inspection_policy.json", policy_path)
        elif policy_path.exists(): policy_path.unlink()
        if policy_path.exists(): os.chown(policy_path, 10001, 10001)
        shutil.copy2(args.backup / "products.json", registry_path)
        os.chown(registry_path, 10001, 10001)
        safe_extract(args.backup / "source.tar.gz", root, allowed_code)
        run(["docker", "tag", pinned, "deploy-app"])
        run(compose + ["up", "-d", "--no-deps", "app"])
        raise


if __name__ == "__main__": main()
