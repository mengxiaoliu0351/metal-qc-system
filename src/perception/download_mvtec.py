# -*- coding: utf-8 -*-
"""并行下载 MVTec 镜像文件（curl 子进程，规避 hf-mirror 对 urllib 的 403）。
用法：cd 到 data 目录后运行：python code/download_mvtec.py
"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MANIFEST = os.path.join(ROOT, "mvtec_files.txt")
OUTDIR = os.path.join(ROOT, "mvtec")


def dl(pair: str):
    dest, url = pair.split("|", 1)
    full = os.path.join(OUTDIR, dest)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    r = subprocess.run(
        ["curl", "-sL", "-A", "Mozilla/5.0", "--max-time", "60", "-o", full, url],
        capture_output=True,
    )
    ok = os.path.exists(full) and os.path.getsize(full) > 100
    if not ok:
        print("FAIL", dest)
    return ok


def main():
    lines = [l.strip() for l in open(MANIFEST, encoding="utf-8") if l.strip()]
    print(f"待下载 {len(lines)} 个文件，8 路并行")
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(dl, lines))
    print(f"成功 {sum(res)} / {len(lines)}")


if __name__ == "__main__":
    main()
