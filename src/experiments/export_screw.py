# -*- coding: utf-8 -*-
"""导出 MVTec parquet (TheoM55 镜像) 为标准目录结构 train/good, test/<defect>"""
import os, sys, io
import pandas as pd
from PIL import Image

SRC = r"D:\programme file\ResearchArticle\data\_dl\screw.train.parquet"
OUT_ROOT = r"D:\programme file\ResearchArticle\data\mvtec\screw"

df = pd.read_parquet(SRC)
print("cols:", list(df.columns), "rows:", len(df))
print("split:", df["split"].value_counts().to_dict())
print("defect:", df["defect"].value_counts().to_dict())
print("label:", df["label"].value_counts().to_dict())
print("image_path type:", type(df.iloc[0]["image_path"]))
print("image_path sample keys:", df.iloc[0]["image_path"].keys() if isinstance(df.iloc[0]["image_path"], dict) else "N/A")

os.makedirs(os.path.join(OUT_ROOT, "train", "good"), exist_ok=True)
n = 0
for i, row in df.iterrows():
    ip = row["image_path"]
    b = ip["bytes"] if isinstance(ip, dict) else None
    if b is None:
        continue
    img = Image.open(io.BytesIO(b)).convert("RGB")
    # train 全是 good
    fn = f"{i:04d}.png"
    img.save(os.path.join(OUT_ROOT, "train", "good", fn))
    n += 1
print(f"导出 train/good: {n} 张 -> {os.path.join(OUT_ROOT,'train','good')}")
