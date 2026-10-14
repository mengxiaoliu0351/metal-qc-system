# -*- coding: utf-8 -*-
"""验证 screw 库在 test 集上的分离度：good 与各缺陷类的异常分数分布"""
import os, sys, io
import numpy as np, pandas as pd
from PIL import Image

ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from perception.winclip_patch import WinCLIPPatch

BANK = os.path.join(ROOT, "results", "perception", "banks", "金属螺丝(screw).npy")
TEST = os.path.join(ROOT, "data", "_dl", "screw.test.parquet")
THR = 0.1788

det = WinCLIPPatch("ViT-B-16")
det.bank = np.load(BANK)
det.bank_info = {"n_images": 80, "n_feats": det.bank.shape[0]}

df = pd.read_parquet(TEST)
print(f"test共 {len(df)} 张, defect类别: {df['defect'].value_counts().to_dict()}")
good, bad = [], []
for _, row in df.iterrows():
    ip = row["image_path"]
    b = ip["bytes"] if isinstance(ip, dict) else None
    if b is None: continue
    img = Image.open(io.BytesIO(b)).convert("RGB")
    s = float(det.detect(img)["score"])
    if row["defect"] == "good":
        good.append(s)
    else:
        bad.append((row["defect"], s))

g = np.array(good)
bs = np.array([s for _, s in bad])
print(f"\n[good] {len(g)}张  均值{g.mean():.4f} 范围[{g.min():.4f},{g.max():.4f}]  误判不合格(>{THR}): {(g>THR).sum()}")
print(f"[缺陷] {len(bs)}张 均值{bs.mean():.4f} 范围[{bs.min():.4f},{bs.max():.4f}]  检出(>{THR}): {(bs>THR).sum()}/{len(bs)}")
# 按缺陷类
from collections import defaultdict
d = defaultdict(list)
for name, s in bad: d[name].append(s)
for k, v in d.items():
    vv = np.array(v)
    print(f"  {k}: n={len(vv)} 均值{vv.mean():.4f} 检出率{(vv>THR).mean()*100:.0f}%")
