# -*- coding: utf-8 -*-
"""为NEU-DET钢带建库（用挑出的18张图作为伪正常样本）"""
import os, sys
ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from PIL import Image
import numpy as np
from perception.winclip_patch import WinCLIPPatch

NORMAL_DIR = os.path.join(ROOT, "data", "neu_normal_pick")
BANK_PATH = os.path.join(ROOT, "results", "perception", "banks", "钢带(NEU-DET).npy")

print("加载CLIP...")
det = WinCLIPPatch("ViT-B-16")
files = [f for f in os.listdir(NORMAL_DIR) if f.lower().endswith((".jpg",".png"))]
print(f"建库：{len(files)}张伪正常图")
feats = []
for i, f in enumerate(files):
    img = Image.open(os.path.join(NORMAL_DIR, f)).convert("RGB")
    feats.append(det._get_patch_feats(img))
    print(f"  {i+1}/{len(files)} {f}")
det.bank = np.concatenate(feats, axis=0)
os.makedirs(os.path.dirname(BANK_PATH), exist_ok=True)
np.save(BANK_PATH, det.bank)
print(f"\n✅ 库已保存：{BANK_PATH}")
print(f"   特征矩阵：{det.bank.shape}")
