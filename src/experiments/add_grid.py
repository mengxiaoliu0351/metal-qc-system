# -*- coding: utf-8 -*-
"""导出 grid parquet -> train/good，建库+独立标定，追加products.json并移除NEU产品项"""
import os, sys, io, json
import numpy as np, pandas as pd
from PIL import Image

ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from perception.winclip_patch import WinCLIPPatch

CATEGORY = "grid"
KEY, NAME = "grid", "金属网格 (MVTec grid)"
BANK_FILE = "金属网格(grid).npy"
PQT = os.path.join(ROOT, "data", "_dl", f"{CATEGORY}.train.parquet")
GOOD_DIR = os.path.join(ROOT, "data", "mvtec", CATEGORY, "train", "good")
BANKS = os.path.join(ROOT, "results", "perception", "banks")
BANK_PATH = os.path.join(BANKS, BANK_FILE)

# 1) 导出 good 图
os.makedirs(GOOD_DIR, exist_ok=True)
df = pd.read_parquet(PQT)
print("cols:", list(df.columns), "rows:", len(df))
print("defect:", df["defect"].value_counts().to_dict())
n = 0
for i, row in df.iterrows():
    ip = row["image_path"]
    b = ip["bytes"] if isinstance(ip, dict) else None
    if b is None: continue
    Image.open(io.BytesIO(b)).convert("RGB").save(os.path.join(GOOD_DIR, f"{i:04d}.png"))
    n += 1
print(f"导出 good: {n} 张")

# 2) 建库（前80）+ 独立标定（holdout）
det = WinCLIPPatch("ViT-B-16")
det.build_bank(GOOD_DIR, max_samples=80, cache_path=BANK_PATH)
bank = np.load(BANK_PATH); print(f"建库: {BANK_FILE} shape={bank.shape}")

files = sorted([f for f in os.listdir(GOOD_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))])
held = files[80:]
print(f"独立标定 {len(held)} 张...")
scores = []
for fn in held:
    img = Image.open(os.path.join(GOOD_DIR, fn)).convert("RGB")
    scores.append(float(det.detect(img)["score"]))
arr = np.array(scores)
median = float(np.median(arr)); mad = float(np.median(np.abs(arr - median)))
thr = round(median + 1.5*mad, 4)
cal = {"threshold": thr, "median": round(median,4), "mad": round(mad,4),
       "n_normal": len(held), "score_min": round(float(arr.min()),4),
       "score_max": round(float(arr.max()),4), "method": "independent-holdout"}
print(f"median={median:.4f} MAD={mad:.4f} 阈值={thr} 分数范围[{arr.min():.4f},{arr.max():.4f}]")

# 3) products.json：移除 neu_det，追加 grid
pj = os.path.join(BANKS, "products.json")
cfg = json.load(open(pj, encoding="utf-8"))
cfg = [c for c in cfg if c["key"] not in ("neu_det", KEY)]
cfg.append({"key": KEY, "name": NAME, "bank": BANK_FILE, "threshold": thr, "calibration": cal})
json.dump(cfg, open(pj,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n最终产品库：")
for c in cfg: print(f"  {c['name']} 阈值={c['threshold']}")
