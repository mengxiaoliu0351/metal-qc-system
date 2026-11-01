"""Build an allowlisted deployment bundle; never include local histories, credentials or datasets."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".preview/aliyun-deploy-20261025"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = set()
    files.add(ROOT / ".dockerignore")
    for folder in (ROOT / "src", ROOT / "deploy"):
        for path in folder.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
                # Experiment programs and legacy UI are not needed by the cloud app.
                if "experiments" not in path.parts and "weights" not in path.parts:
                    files.add(path)
    products_path = ROOT / "results/perception/banks/products.json"
    products = json.loads(products_path.read_text(encoding="utf-8"))
    files.add(products_path)
    for product in products:
        files.add(products_path.parent / product["bank"])
        classifier = ROOT / f"results/perception/classifier_{product['key']}.joblib"
        if classifier.exists():
            files.add(classifier)
        meta = ROOT / "results/perception/optimized" / product["perception_artifact"]
        files.add(meta)
        files.add(meta.parent / json.loads(meta.read_text(encoding="utf-8"))["arrays"])
    csv = ROOT / "results/cognition/standard_library.csv"
    if csv.exists():
        files.add(csv)
    # Weights are uploaded once separately, avoiding repeated 600MB bundles during fixes.
    manifest = []
    archive = OUT / "metalqc-bundle.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=3) as bundle:
        for path in sorted(files):
            relative = path.relative_to(ROOT).as_posix()
            if path.name.startswith(".env") or path.suffix in (".pem", ".key"):
                raise ValueError("Credential file in bundle")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            bundle.add(path, arcname=relative)
            manifest.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    (OUT / "bundle-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "files": len(files)}))


if __name__ == "__main__":
    main()
