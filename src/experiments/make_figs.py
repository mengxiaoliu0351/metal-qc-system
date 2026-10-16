# -*- coding: utf-8 -*-
"""生成方法选型对比图（用于中期报告）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:\programme file\ResearchArticle\results\experiments"

# ===== 图1：距离度量AUC对比（方法选型实验）=====
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
methods = ["最近邻余弦\n(本系统采用)", "kNN-5余弦", "PaDiM马氏\n(PCA降维)"]
aucs = [98.68, 98.24, 62.66]
colors = ["#2e7d32", "#90a4ae", "#c62828"]
bars = ax.bar(methods, aucs, color=colors, width=0.55)
for b, v in zip(bars, aucs):
    ax.text(b.get_x()+b.get_width()/2, v+0.8, f"{v:.2f}%", ha="center", fontsize=12, fontweight="bold")
ax.set_ylim(50, 105)
ax.set_ylabel("AUROC (%)")
ax.set_title("异常检测距离度量对比（MVTec metal_nut，115张）", fontsize=12, fontweight="bold")
ax.axhline(98.68, ls="--", color="#2e7d32", alpha=0.4)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + r"\fig_metric_comparison.png", bbox_inches="tight")
plt.close()

# ===== 图2：三方案对比雷达式柱状（准确率/速度/VLM调用率）=====
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
labels = ["准确率(%)", "推理速度(0.1x)", "VLM调用率(100-x)"]
winclip = [94.78, 7, 100]
vlm = [90.0, 1, 0]
dual = [96.52, 7.7, 81]   # 双链路：96.52%、快0.227s(vs VLM 0.9s≈4倍→7.7)、VLM调用19%→81%免调
x = np.arange(len(labels)); w = 0.26
ax.bar(x-w, winclip, w, label="纯WinCLIP", color="#90a4ae")
ax.bar(x, vlm, w, label="纯VLM", color="#ffb74d")
ax.bar(x+w, dual, w, label="双链路门控(本系统)", color="#1565c0")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_title("纯WinCLIP / 纯VLM / 双链路 三方对比", fontsize=12, fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + r"\fig_three_way.png", bbox_inches="tight")
plt.close()
print("图已保存到", OUT)
