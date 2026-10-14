# -*- coding: utf-8 -*-
"""NEU-DET同域测试：用NEU自己建的库测NEU缺陷图，看分数分布"""
import os, sys, json
import numpy as np
from PIL import Image
ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT,"src"))
from perception.winclip_patch import WinCLIPPatch

det = WinCLIPPatch("ViT-B-16")
det.bank = np.load(os.path.join(ROOT,"results","perception","banks","钢带(NEU-DET).npy"))
det.bank_info = {"n_images":18, "n_feats":det.bank.shape[0]}

NEU_DIR = os.path.join(ROOT,"data","NEU-DET","IMAGES")
cats = ["crazing","inclusion","pitted_surface","patches","rolled-in_scale","scratches"]
records = []
for cat in cats:
    files = sorted([f for f in os.listdir(NEU_DIR) if f.startswith(cat+"_")])[:20]
    for fn in files:
        img = Image.open(os.path.join(NEU_DIR, fn)).convert("RGB")
        r = det.detect(img)
        records.append({"cat":cat, "file":fn, "score":float(r["score"])})

scores = [r["score"] for r in records]
print(f"NEU-DET同域测试（NEU库测NEU图）：{len(records)}张")
print(f"分数均值：{np.mean(scores):.4f}")
print(f"分数中位数：{np.median(scores):.4f}")
print(f"分数范围：[{np.min(scores):.4f}, {np.max(scores):.4f}]")
print()
for cat in cats:
    sub = [r["score"] for r in records if r["cat"]==cat]
    print(f"  {cat}: mean={np.mean(sub):.4f} median={np.median(sub):.4f}")

# 对比跨域结果（之前用metal_nut库测NEU）
print("\n--- 对比 ---")
print(f"跨域（metal_nut库测NEU）：均值0.4804，100%超过metal_nut阈值0.2243")
print(f"同域（NEU库测NEU）：均值{np.mean(scores):.4f}")
