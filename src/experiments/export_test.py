# -*- coding: utf-8 -*-
"""把 screw / grid 的 test parquet 导出为标准目录 data\mvtec\<cat>\test\<defect>\*.png"""
import os, io
import pandas as pd
from PIL import Image

ROOT = r"D:\programme file\ResearchArticle"
for cat in ("screw", "grid"):
    pqt = os.path.join(ROOT, "data", "_dl", f"{cat}.test.parquet")
    out_root = os.path.join(ROOT, "data", "mvtec", cat, "test")
    df = pd.read_parquet(pqt)
    counts = {}
    for _, row in df.iterrows():
        ip = row["image_path"]
        b = ip["bytes"] if isinstance(ip, dict) else None
        if b is None: continue
        defect = str(row["defect"])
        d = os.path.join(out_root, defect)
        os.makedirs(d, exist_ok=True)
        img = Image.open(io.BytesIO(b)).convert("RGB")
        fn = f"{len([f for f in os.listdir(d) if f.endswith('.png')]):04d}.png"
        img.save(os.path.join(d, fn))
        counts[defect] = counts.get(defect, 0) + 1
    print(f"[{cat}] -> {out_root}")
    for k, v in counts.items():
        print(f"   {k}: {v}张")
