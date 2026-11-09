"""Apply a verified code-only correction, preserving all live data and policy."""
import argparse
import json
from pathlib import Path
import tarfile
import time
import urllib.request

from activate_engineering import checksum, run, safe_extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    root = Path("/opt/metalqc")
    stage, backup = args.stage.resolve(), args.backup.resolve()
    if not stage.is_relative_to(Path("/opt/metalqc-staging")) or not backup.is_relative_to(Path("/opt/metalqc-backups")):
        raise ValueError("Unexpected update location")
    if backup.exists(): raise ValueError("Backup already exists; inspect prior update before retrying")
    manifest = json.loads((stage / "manifest.json").read_text())
    if checksum(stage / "code.tar.gz") != manifest["code.tar.gz"]["sha256"]:
        raise ValueError("Code checksum mismatch")
    compose = ["docker", "compose", "-f", str(root / "deploy/compose.yaml")]
    container = run(compose + ["ps", "-q", "app"], capture=True).stdout.strip()
    image = run(["docker", "inspect", "--format", "{{.Image}}", container], capture=True).stdout.strip()
    pinned = "metalqc-backup:" + backup.name
    run(["docker", "tag", image, pinned])
    backup.mkdir(mode=0o700, parents=True)
    with tarfile.open(backup / "source.tar.gz", "w:gz", compresslevel=1) as archive:
        for relative in ("src", "deploy/Dockerfile", "deploy/requirements-cloud.txt"):
            archive.add(root / relative, arcname=relative, filter=lambda m: None if "__pycache__" in m.name else m)
    allowed = lambda n: n.startswith("src/") or n in ("src", "deploy", "deploy/Dockerfile", "deploy/requirements-cloud.txt")
    try:
        safe_extract(stage / "code.tar.gz", root, allowed)
        run(compose + ["build", "app"])
        run(compose + ["up", "-d", "--no-deps", "app"])
        for _ in range(30):
            try:
                with urllib.request.urlopen("http://127.0.0.1:18000/healthz", timeout=5) as response:
                    if json.load(response).get("ok"): break
            except Exception: time.sleep(2)
        else: raise RuntimeError("Updated app failed health checks")
        receipt = {"healthy": True, "code_sha256": manifest["code.tar.gz"]["sha256"], "rollback_image": pinned}
        (backup / "deployment.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)
    except Exception:
        safe_extract(backup / "source.tar.gz", root, allowed)
        run(["docker", "tag", pinned, "deploy-app"])
        run(compose + ["up", "-d", "--no-deps", "app"])
        raise


if __name__ == "__main__": main()
