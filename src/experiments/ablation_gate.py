# -*- coding: utf-8 -*-
"""
消融实验：单阈值 vs 三区间门控
================================
用 exp1 已存的 metal_nut 115 张测试集分数，不重跑模型。
t_low=0.2345 (median+1.5MAD 标定), t_high=0.3518 (启发式 t_low*1.5，暂定)。
"""
import os, sys, json
import numpy as np

ROOT = r"D:\programme file\ResearchArticle"
scores_path = os.path.join(ROOT, "results", "experiments", "exp1_winclip_scores.json")
recs = json.load(open(scores_path, encoding="utf-8"))

T_LOW = 0.2345
T_HIGH = 0.30  # 基于正常件分数分布(max=0.237)与缺陷中位数(0.305)的工程折中

scores = np.array([r["score"] for r in recs])
labels = np.array([r["label"] for r in recs])  # 0=正常 1=异常

# ===== 1. 单阈值（旧设计：s>=t_low 判异常）=====
preds_single = (scores >= T_LOW).astype(int)
fp_single = int(((preds_single == 1) & (labels == 0)).sum())  # 正常判异常
fn_single = int(((preds_single == 0) & (labels == 1)).sum())  # 异常判正常
tp_single = int(((preds_single == 1) & (labels == 1)).sum())
tn_single = int(((preds_single == 0) & (labels == 0)).sum())

# ===== 2. 三区间（新设计）=====
low = scores < T_LOW
high = scores > T_HIGH
gray = ~low & ~high  # 灰区 [t_low, t_high]

# 低风险区：本地放行（判正常）
low_fp = int((low & (labels == 0)).sum())   # 正常在低风险（正确放行）
low_fn = int((low & (labels == 1)).sum())   # 异常在低风险（漏检！）

# 高风险区：本地提示异常
high_tp = int((high & (labels == 1)).sum())  # 异常在高风险（正确提示）
high_fp = int((high & (labels == 0)).sum()) # 正常在高风险（误报！）

# 灰区：调VLM
gray_n = int(gray.sum())
gray_normal = int((gray & (labels == 0)).sum())
gray_defect = int((gray & (labels == 1)).sum())

# 三区间"理想VLM"（假设灰区VLM100%正确）：
# 误报 = 高风险中正常件 + 灰区中正常件（VLM也会误判？假设正确=0）
# 漏检 = 低风险中异常件（VLM不管）
# 保守估计：灰区正常件即使VLM复核也可能漏报，灰区异常件VLM能捞回
fp_ideal = high_fp  # VLM正确复核灰区，灰区正常件不进异常
fn_ideal = low_fn  # VLM捞回灰区异常件，只剩低风险漏检

# VLM调用率
vlm_rate = gray_n / len(recs)

print("=" * 60)
print(f"测试集: {len(recs)} 张 (正常 {(labels==0).sum()} + 缺陷 {(labels==1).sum()})")
print(f"t_low={T_LOW}, t_high={T_HIGH} (启发式暂定)")
print("=" * 60)

print("\n【1. 单阈值门控（旧设计）】")
print(f"  混淆矩阵: TP={tp_single} TN={tn_single} FP={fp_single} FN={fn_single}")
print(f"  误报(正常判异常): {fp_single}")
print(f"  漏检(异常判正常): {fn_single}")
print(f"  VLM调用率: 0% (所有图本地判)")

print("\n【2. 三区间门控（新设计）】")
print(f"  低风险区(s<{T_LOW}): {int(low.sum())}张 (正常{low_fp} + 缺陷{low_fn})")
print(f"  灰区({T_LOW}<=s<={T_HIGH}): {gray_n}张 (正常{gray_normal} + 缺陷{gray_defect})")
print(f"  高风险区(s>{T_HIGH}): {int(high.sum())}张 (正常{high_fp} + 缺陷{high_tp})")
print(f"  VLM调用率: {vlm_rate*100:.1f}% (仅灰区{gray_n}张调VLM)")

print("\n【3. 对比：误报/漏检】")
print(f"  单阈值: 误报{fp_single} 漏检{fn_single}")
print(f"  三区间(理想VLM复核灰区): 误报{fp_ideal} 漏检{fn_ideal}")
print(f"  三区间(不调VLM, 灰区一律放行): 误报{high_fp} 漏检{fn_ideal+gray_defect}")

# 保存结果
out = {
    "t_low": T_LOW, "t_high": T_HIGH, "n_total": len(recs),
    "n_normal": int((labels==0).sum()), "n_defect": int((labels==1).sum()),
    "single_threshold": {"TP": tp_single, "TN": tn_single, "FP": fp_single, "FN": fn_single},
    "three_zone": {
        "low_risk": int(low.sum()), "gray": gray_n, "high_risk": int(high.sum()),
        "gray_normal": gray_normal, "gray_defect": gray_defect,
        "high_fp": high_fp, "low_fn": low_fn, "high_tp": high_tp,
        "vlm_call_rate_pct": round(vlm_rate*100, 1),
    },
    "ideal_vlm": {"FP": fp_ideal, "FN": fn_ideal},
}
out_path = os.path.join(ROOT, "results", "experiments", "ablation_gate.json")
json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n结果已保存: {out_path}")
