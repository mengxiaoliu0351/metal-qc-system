# -*- coding: utf-8 -*-
"""预生成演示样例缓存：感知层分数 + 热力图 + 门控路由"""
import os, sys, json
ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from PIL import Image
from perception.winclip_patch import WinCLIPPatch
from gating.router import gate
from config import SAMPLE_DIR, RESULTS_DIR, THRESHOLD

det = WinCLIPPatch("ViT-B-16")
bank_cache = os.path.join(RESULTS_DIR, "perception", "memory_bank_80.npy")
det.build_bank(os.path.join(ROOT, "data", "mvtec", "metal_nut", "train", "good"),
               80, bank_cache)
print("记忆库已加载（缓存）")

cache_root = os.path.join(RESULTS_DIR, "demo_cache")
os.makedirs(cache_root, exist_ok=True)

for name in sorted(os.listdir(SAMPLE_DIR)):
    src = os.path.join(SAMPLE_DIR, name)
    stem = os.path.splitext(name)[0]
    out_dir = os.path.join(cache_root, stem)
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(src).convert("RGB")
    r = det.detect(img)
    g = gate(r["score"], THRESHOLD)

    input_png = os.path.join(out_dir, "input.png")
    heat_png = os.path.join(out_dir, "heatmap.png")
    img.save(input_png)
    r["heatmap"].save(heat_png)

    result = {
        "input_png": input_png, "heatmap_png": heat_png,
        "score": r["score"], "gate": g,
        "note": "预缓存样例（感知层真实推理）",
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"{name:<26} score={r['score']:.4f} -> {g['route']} 已缓存")
