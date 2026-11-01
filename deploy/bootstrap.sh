#!/usr/bin/env bash
set -euo pipefail
cd /opt/metalqc
tar -xzf uploads/metalqc-bundle.tar.gz
tar -xzf uploads/metalqc-benchmark.tar.gz
install -m 644 uploads/CLIP-ViT-B-16-open_clip_model.safetensors weights/CLIP-ViT-B-16-open_clip_model.safetensors
chmod 755 weights
if [ ! -f runtime.env ]; then
  umask 077
  python3 - <<'PY'
from pathlib import Path
import secrets
Path('runtime.env').write_text('INSPECTION_ADMIN_TOKEN=' + secrets.token_hex(32) + '\nDASHSCOPE_API_KEY=\n', encoding='utf-8')
PY
fi
chmod 600 runtime.env
python3 deploy/prepare_demo_login.py
chown -R 10001:10001 results
docker compose -f deploy/compose.yaml build
docker compose -f deploy/compose.yaml up -d
docker compose -f deploy/compose.yaml ps
