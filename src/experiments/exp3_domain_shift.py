# -*- coding: utf-8 -*-
"""
域偏移测试：metal_nut记忆库 -> NEU-DET钢带图
=============================================
不重新建记忆库，直接用metal_nut的80张正常patch去测NEU-DET的缺陷图。
目的：展示跨域分数漂移，说明本系统"针对特定产品建库"的必要性。
"""
import os, sys, json, time
import numpy as np
from PIL import Image

ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "perception"))

from perception.winclip_patch import WinCLIPPatch
from config import THRESHOLD

NEU_DIR = os.path.join(ROOT, "data", "NEU-DET", "IMAGES")
OUT_DIR = os.path.join(ROOT, "results", "experiments")

def main():
    print("加载WinCLIP（用metal_nut记忆库）...")
    det = WinCLIPPatch("ViT-B-16")
    bank = os.path.join(ROOT, "results", "perception", "memory_bank_80.npy")
    det.build_bank(os.path.join(ROOT, "data", "mvtec", "metal_nut", "train", "good"),
                   max_samples=80, cache_path=bank)

    # 6类缺陷各抽20张，共120张
    cats = ["crazing", "inclusion", "pitted_surface", "patches", "rolled-in_scale", "scratches"]
    records = []
    for cat in cats:
        files = sorted([f for f in os.listdir(NEU_DIR) if f.startswith(cat+"_")])[:20]
        print(f"\n[{cat}] {len(files)}张")
        for i, fn in enumerate(files):
            try:
                img = Image.open(os.path.join(NEU_DIR, fn)).convert("RGB")
                r = det.detect(img)
                records.append({"category": cat, "file": fn, "score": float(r["score"])})
            except Exception as e:
                print(f"  {fn} 失败: {e}")
            if (i+1) % 10 == 0:
                print(f"  {i+1}/{len(files)}")

    with open(os.path.join(OUT_DIR, "exp3_domain_shift.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    scores = np.array([r["score"] for r in records])
    print("\n" + "="*50)
    print(f"NEU-DET跨域测试：{len(records)}张")
    print(f"分数均值：{scores.mean():.4f}")
    print(f"分数中位数：{np.median(scores):.4f}")
    print(f"分数范围：[{scores.min():.4f}, {scores.max():.4f}]")
    print(f"metal_nut阈值：{THRESHOLD}")
    print(f"超过阈值的比例：{(scores>=THRESHOLD).mean()*100:.1f}%")
    print("="*50)
    for cat in cats:
        sub = [r["score"] for r in records if r["category"]==cat]
        if sub:
            print(f"  {cat}: mean={np.mean(sub):.4f} median={np.median(sub):.4f}")

if __name__ == "__main__":
    main()
