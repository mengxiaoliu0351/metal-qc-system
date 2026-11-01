"""Package only the existing frozen benchmark images and matching feature caches."""
import json
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
benchmark = root / 'results/experiments/perception_upgrade_v1'
files = set()
for product in ('metal_nut', 'screw', 'grid'):
    split = benchmark / (product + '_split.json')
    files.update((split, benchmark / (product + '_features.npz')))
    for value in json.loads(split.read_text(encoding='utf-8')).values():
        if isinstance(value, list):
            files.update(root / name.replace('\\', '/') for name in value)
output = root / '.preview/aliyun-deploy-20261025/metalqc-benchmark.tar.gz'
with tarfile.open(output, 'w:gz', compresslevel=1) as archive:
    for path in sorted(files):
        archive.add(path, arcname=path.relative_to(root).as_posix())
print(json.dumps({'files': len(files), 'bytes': output.stat().st_size}))
