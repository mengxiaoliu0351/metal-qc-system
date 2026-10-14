# -*- coding: utf-8 -*-
"""为 screw（金属螺丝）建库并用独立合格样本标定阈值，追加进 products.json"""
import os, sys, json
import numpy as np
from PIL import Image

ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from perception.winclip_patch import WinCLIPPatch

BANKS = os.path.join(ROOT, "results", "perception", "banks")
GOOD_DIR = os.path.join(ROOT, "data", "mvtec", "screw", "train", "good")
BANK_FILE = "金属螺丝(screw).npy"
BANK_PATH = os.path.join(BANKS, BANK_FILE)
KEY, NAME = "screw", "金属螺丝 (MVTec screw)"

det = WinCLIPPatch("ViT-B-16")

# 1) 建库：用前 80 张合格件（与 metal_nut 一致的规模）
det.build_bank(GOOD_DIR, max_samples=80, cache_path=BANK_PATH)
bank = np.load(BANK_PATH)
print(f"建库完成: {BANK_FILE} shape={bank.shape}")

# 2) 独立标定：用第 81 张起的合格件（不进库）算 median+1.5*MAD
files = sorted([f for f in os.listdir(GOOD_DIR)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))])
held = files[80:]
print(f"独立标定样本 {len(held)} 张（与建库的前80张不重叠）...")
scores = []
for fn in held:
    img = Image.open(os.path.join(GOOD_DIR, fn)).convert("RGB")
    scores.append(float(det.detect(img)["score"]))
arr = np.array(scores)
median = float(np.median(arr)); mad = float(np.median(np.abs(arr - median)))
thr = round(median + 1.5 * mad, 4)
cal = {"threshold": thr, "median": round(median, 4), "mad": round(mad, 4),
       "n_normal": len(held), "score_min": round(float(arr.min()), 4),
       "score_max": round(float(arr.max()), 4), "method": "independent-holdout"}
print(f"median={median:.4f} MAD={mad:.4f} 阈值={thr}  分数范围[{arr.min():.4f},{arr.max():.4f}]")

# 3) 追加进 products.json
pj = os.path.join(BANKS, "products.json")
cfg = json.load(open(pj, encoding="utf-8"))
cfg = [c for c in cfg if c["key"] != KEY]
cfg.append({"key": KEY, "name": NAME, "bank": BANK_FILE,
            "threshold": thr, "calibration": cal})
json.dump(cfg, open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("products.json 现有产品：")
for c in cfg:
    print(f"  {c['name']} 阈值={c['threshold']}")
