# -*- coding: utf-8 -*-
"""为每个记忆库标定独立阈值，并生成 products.json 产品配置"""
import os, sys, json
import numpy as np
from PIL import Image

ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from perception.winclip_patch import WinCLIPPatch

BANKS_DIR = os.path.join(ROOT, "results", "perception", "banks")

# 产品配置：产品名 -> (库文件名, 正常样本目录, 说明)
PRODUCTS = [
    {
        "key": "metal_nut",
        "name": "金属螺母 (MVTec metal_nut)",
        "bank": "金属螺母(metal_nut).npy",
        "normal_dir": os.path.join(ROOT, "data", "mvtec", "metal_nut", "train", "good"),
        "threshold": 0.2345,  # 已标定
    },
    {
        "key": "neu_det",
        "name": "热轧钢带 (NEU-DET)",
        "bank": "钢带(NEU-DET).npy",
        "normal_dir": os.path.join(ROOT, "data", "neu_normal_pick"),
        "threshold": None,  # 待标定
    },
]


def calibrate(det, normal_dir, max_samples=80):
    """在正常样本上算 median + 1.5*MAD 阈值"""
    files = sorted([f for f in os.listdir(normal_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])[:max_samples]
    scores = []
    for fn in files:
        img = Image.open(os.path.join(normal_dir, fn)).convert("RGB")
        r = det.detect(img)
        scores.append(float(r["score"]))
    arr = np.array(scores)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    thr = median + 1.5 * mad
    return {
        "threshold": round(thr, 4),
        "median": round(median, 4),
        "mad": round(mad, 4),
        "n_normal": len(files),
        "score_min": round(float(arr.min()), 4),
        "score_max": round(float(arr.max()), 4),
    }


def calibrate_loo(det, normal_dir):
    """留一法标定：正常样本同时用于建库时，每次排除被测图，避免数据泄露。
    返回正常分数分布与 median+1.5*MAD 阈值。"""
    files = sorted([f for f in os.listdir(normal_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"  提取 {len(files)} 张正常图的patch特征...")
    feats_per_img = []
    images = []
    for fn in files:
        img = Image.open(os.path.join(normal_dir, fn)).convert("RGB")
        images.append(img)
        feats_per_img.append(det._get_patch_feats(img))

    scores = []
    n = len(files)
    for i in range(n):
        # 用除第i张外的所有图建库
        bank = np.concatenate([feats_per_img[j] for j in range(n) if j != i], axis=0)
        det.bank = bank
        det.bank_info = {"n_images": n - 1, "n_feats": bank.shape[0]}
        r = det.detect(images[i])
        scores.append(float(r["score"]))
    arr = np.array(scores)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    thr = median + 1.5 * mad
    return {
        "threshold": round(thr, 4),
        "median": round(median, 4),
        "mad": round(mad, 4),
        "n_normal": n,
        "score_min": round(float(arr.min()), 4),
        "score_max": round(float(arr.max()), 4),
        "method": "leave-one-out",
    }


def main():
    det = WinCLIPPatch("ViT-B-16")
    config = []
    for p in PRODUCTS:
        bank_path = os.path.join(BANKS_DIR, p["bank"])
        det.bank = np.load(bank_path)
        det.bank_info = {"n_images": 80, "n_feats": det.bank.shape[0]}
        cal = None
        if p["threshold"] is None and os.path.isdir(p["normal_dir"]):
            print(f"标定 {p['name']}（留一法，正常样本即建库样本）...")
            cal = calibrate_loo(det, p["normal_dir"])
            p["threshold"] = cal["threshold"]
            print(f"  正常样本{cal['n_normal']}张, median={cal['median']}, "
                  f"MAD={cal['mad']}, 阈值={cal['threshold']}")
        elif os.path.isdir(p["normal_dir"]):
            print(f"复核 {p['name']}（独立正常样本）...")
            cal = calibrate(det, p["normal_dir"])
            print(f"  正常样本{cal['n_normal']}张, 阈值={cal['threshold']}（沿用 {p['threshold']}）")
            cal["threshold"] = p["threshold"]
        config.append({
            "key": p["key"],
            "name": p["name"],
            "bank": p["bank"],
            "threshold": p["threshold"],
            "calibration": cal,
        })

    out_path = os.path.join(BANKS_DIR, "products.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\n产品配置已保存: {out_path}")
    for c in config:
        print(f"  {c['name']}: 阈值={c['threshold']}")


if __name__ == "__main__":
    main()
