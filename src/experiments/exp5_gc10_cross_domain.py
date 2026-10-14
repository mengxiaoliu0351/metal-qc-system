# -*- coding: utf-8 -*-
"""GC10-DET 域偏移测试
======================
用 metal_nut 记忆库测 GC10-DET 钢板缺陷图，验证跨产品域偏移。
GC10-DET 含 10 类钢板表面缺陷、约2300张灰度图。

数据获取：
  Kaggle: https://www.kaggle.com/zhangyunsheng/defects-class-and-location
  GitHub: https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets
  百度网盘: https://pan.baidu.com/s/1uli6KaAhJ-ZZvon-n3xVZQ  验证码 wsht

下载后将图片放到 data/GC10-DET/IMAGES/ 下（按类别分子目录或文件名前缀均可）。
"""
import os, sys, json
import numpy as np
from PIL import Image

ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from perception.winclip_patch import WinCLIPPatch

# ---------- 配置 ----------
GC10_DIR = os.path.join(ROOT, "data", "GC10-DET", "IMAGES")
THRESHOLD = 0.2345  # metal_nut 阈值（median + 1.5×MAD）
N_PER_CAT = 20      # 每类抽样数量

# GC10 十类缺陷（实际文件夹名 -> 中文名）
CATEGORIES = {
    "punching_hole": "冲孔(Punching)",
    "welding_line": "焊缝(Weld line)",
    "crescent_gap": "新月缝(Crescent gap)",
    "water_spot": "水斑(Water spot)",
    "oil_spot": "油斑(Oil spot)",
    "silk_spot": "丝斑(Silk spot)",
    "inclusion": "夹杂(Inclusion)",
    "rolled_pit": "轧坑(Rolling pit)",
    "crease": "折痕(Crease)",
    "waist folding": "折叠(Folding)",
}


def load_gc10_images():
    """加载GC10图片，按类别子目录组织"""
    samples = []
    if not os.path.isdir(GC10_DIR):
        return samples
    for code, name in CATEGORIES.items():
        subdir = os.path.join(GC10_DIR, code)
        if os.path.isdir(subdir):
            files = sorted([f for f in os.listdir(subdir)
                           if f.lower().endswith((".jpg", ".png", ".bmp"))])[:N_PER_CAT]
            for fn in files:
                samples.append((code, name, os.path.join(subdir, fn)))
    return samples


def main():
    # 加载 metal_nut 记忆库
    det = WinCLIPPatch("ViT-B-16")
    bank_path = os.path.join(ROOT, "results", "perception", "banks",
                             "金属螺母(metal_nut).npy")
    if not os.path.exists(bank_path):
        print(f"[错误] 记忆库不存在: {bank_path}")
        return
    det.bank = np.load(bank_path)
    det.bank_info = {"n_images": 80, "n_feats": det.bank.shape[0]}
    print(f"已加载 metal_nut 记忆库: {det.bank.shape}")

    samples = load_gc10_images()
    if not samples:
        print(f"\n[提示] 未找到GC10图片，请将数据集放到: {GC10_DIR}")
        print("支持两种目录结构：")
        print("  1. data/GC10-DET/IMAGES/Pu/xxx.jpg（按类别分子目录）")
        print("  2. data/GC10-DET/IMAGES/Pu_001.jpg（文件名前缀）")
        return

    print(f"共加载 {len(samples)} 张GC10图片，开始跨域检测...\n")

    # 逐张检测
    records = []
    for i, (code, name, path) in enumerate(samples):
        try:
            img = Image.open(path).convert("RGB")
            r = det.detect(img)
            score = float(r["score"])
            over = score >= THRESHOLD
            records.append({
                "cat_code": code, "cat_name": name,
                "file": os.path.basename(path),
                "score": score, "over_threshold": over,
            })
            if (i + 1) % 10 == 0:
                print(f"  已检测 {i+1}/{len(samples)}")
        except Exception as e:
            print(f"  [跳过] {path}: {e}")

    if not records:
        print("没有成功检测的图片。")
        return

    # ---------- 统计 ----------
    scores = [r["score"] for r in records]
    n_over = sum(1 for r in records if r["over_threshold"])
    total = len(records)

    print("\n" + "=" * 60)
    print("GC10-DET 跨域偏移测试结果（metal_nut 库 → GC10 钢板）")
    print("=" * 60)
    print(f"测试图片数：{total}")
    print(f"门控阈值：{THRESHOLD}")
    print(f"异常分数均值：{np.mean(scores):.4f}")
    print(f"异常分数中位数：{np.median(scores):.4f}")
    print(f"分数范围：[{np.min(scores):.4f}, {np.max(scores):.4f}]")
    print(f"超过阈值数量：{n_over}/{total}（{n_over/total*100:.1f}%）")
    print()

    print("分类别统计：")
    print(f"  {'类别':<22} {'数量':>4} {'均值':>8} {'中位数':>8} {'超阈值率':>8}")
    for code, name in CATEGORIES.items():
        sub = [r for r in records if r["cat_code"] == code]
        if not sub:
            continue
        sub_scores = [r["score"] for r in sub]
        sub_over = sum(1 for r in sub if r["over_threshold"])
        print(f"  {name:<22} {len(sub):>4} {np.mean(sub_scores):>8.4f} "
              f"{np.median(sub_scores):>8.4f} {sub_over/len(sub)*100:>7.1f}%")

    # 结论
    print("\n" + "-" * 60)
    over_rate = n_over / total
    if over_rate > 0.9:
        print("结论：GC10钢板图几乎全部超过螺母阈值，跨产品域偏移显著。")
        print("这验证了系统需要「换产建库」——不同产品必须用各自正常样本建库。")
    elif over_rate > 0.5:
        print("结论：超过半数GC10图超过螺母阈值，存在明显域偏移。")
    else:
        print("结论：GC10跨域超阈值比例不高，域偏移不明显（需检查数据）。")

    # 保存结果
    out_path = os.path.join(ROOT, "results", "experiments", "exp5_gc10_cross_domain.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "GC10-DET cross-domain (metal_nut bank)",
            "threshold": THRESHOLD,
            "n_total": total,
            "n_over_threshold": n_over,
            "over_rate": over_rate,
            "score_mean": float(np.mean(scores)),
            "score_median": float(np.median(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "records": records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
