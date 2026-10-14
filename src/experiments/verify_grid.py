# -*- coding: utf-8 -*-
"""验证 grid 库 test 分离度"""
import os, sys, io
import numpy as np, pandas as pd
from PIL import Image
ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from perception.winclip_patch import WinCLIPPatch
BANK = os.path.join(ROOT,"results","perception","banks","金属网格(grid).npy")
TEST = os.path.join(ROOT,"data","_dl","grid.test.parquet")
THR = 0.2311
det = WinCLIPPatch("ViT-B-16"); det.bank = np.load(BANK)
det.bank_info = {"n_images":80,"n_feats":det.bank.shape[0]}
df = pd.read_parquet(TEST)
print("defect:", df["defect"].value_counts().to_dict())
good, bad = [], []
for _, row in df.iterrows():
    ip = row["image_path"]; b = ip["bytes"] if isinstance(ip,dict) else None
    if b is None: continue
    img = Image.open(io.BytesIO(b)).convert("RGB")
    s = float(det.detect(img)["score"])
    (good if row["defect"]=="good" else bad).append(s)
g=np.array(good); bs=np.array(bad)
print(f"[good] n={len(g)} 均值{g.mean():.4f} 范围[{g.min():.4f},{g.max():.4f}] 误判不合格(>{THR}): {(g>THR).sum()}")
print(f"[缺陷] n={len(bs)} 均值{bs.mean():.4f} 范围[{bs.min():.4f},{bs.max():.4f}] 检出(>{THR}): {(bs>THR).sum()}/{len(bs)}")
