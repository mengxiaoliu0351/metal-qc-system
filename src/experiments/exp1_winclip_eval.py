# -*- coding: utf-8 -*-
"""
对比实验1：纯WinCLIP 在 MVTec metal_nut 测试集上评估
====================================================
输出：
- 每类异常的分数分布
- 全局AUROC、准确率（阈值0.2243）
- 快链路/慢链路分流统计
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

TEST_DIR = os.path.join(ROOT, "data", "mvtec", "metal_nut", "test")
OUT_DIR = os.path.join(ROOT, "results", "experiments")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("加载WinCLIP...")
    det = WinCLIPPatch("ViT-B-16")
    bank = os.path.join(ROOT, "results", "perception", "memory_bank_80.npy")
    det.build_bank(os.path.join(ROOT, "data", "mvtec", "metal_nut", "train", "good"),
                   max_samples=80, cache_path=bank)

    records = []
    categories = sorted(os.listdir(TEST_DIR))
    for cat in categories:
        cat_dir = os.path.join(TEST_DIR, cat)
        if not os.path.isdir(cat_dir):
            continue
        files = sorted([f for f in os.listdir(cat_dir) if f.endswith(".png")])
        print(f"\n[{cat}] {len(files)}张")
        for i, fn in enumerate(files):
            img = Image.open(os.path.join(cat_dir, fn)).convert("RGB")
            t0 = time.time()
            r = det.detect(img)
            dt = time.time() - t0
            score = float(r["score"])
            label = 0 if cat == "good" else 1  # 0正常 1异常
            route = "fast" if score >= THRESHOLD else "slow"
            records.append({
                "category": cat, "file": fn, "label": label,
                "score": score, "route": route, "time_s": round(dt, 3),
            })
            if (i+1) % 10 == 0:
                print(f"  {i+1}/{len(files)}  score={score:.4f}")

    # 保存明细
    with open(os.path.join(OUT_DIR, "exp1_winclip_scores.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # 统计
    scores = np.array([r["score"] for r in records])
    labels = np.array([r["label"] for r in records])
    # AUROC
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
    auroc = roc_auc_score(labels, scores)
    preds = (scores >= THRESHOLD).astype(int)  # 分数高=异常
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    # 每类统计
    cat_stats = {}
    for cat in categories:
        sub = [r for r in records if r["category"] == cat]
        s = np.array([r["score"] for r in sub])
        cat_stats[cat] = {
            "n": len(sub),
            "mean": round(float(s.mean()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "fast": sum(1 for r in sub if r["route"]=="fast"),
            "slow": sum(1 for r in sub if r["route"]=="slow"),
        }
    fast_n = sum(1 for r in records if r["route"]=="fast")
    slow_n = sum(1 for r in records if r["route"]=="slow")

    summary = {
        "threshold": THRESHOLD,
        "total": len(records),
        "auroc": round(float(auroc), 4),
        "accuracy": round(float(acc), 4),
        "confusion_matrix": cm.tolist(),
        "fast_route": fast_n, "slow_route": slow_n,
        "avg_time_s": round(float(np.mean([r["time_s"] for r in records])), 3),
        "by_category": cat_stats,
    }
    with open(os.path.join(OUT_DIR, "exp1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print(f"AUROC: {auroc:.4f}")
    print(f"准确率(阈值{THRESHOLD}): {acc:.4f}")
    print(f"混淆矩阵: {cm.tolist()}")
    print(f"快链路: {fast_n}张, 慢链路: {slow_n}张")
    print(f"平均推理: {np.mean([r['time_s'] for r in records]):.3f}s/张")
    print("="*50)
    for cat, s in cat_stats.items():
        print(f"  {cat}: n={s['n']} mean={s['mean']} fast={s['fast']} slow={s['slow']}")

if __name__ == "__main__":
    main()
