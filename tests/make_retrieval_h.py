# -*- coding: utf-8 -*-
"""重做 RAG 检索图为横向布局（适配第11页卡片）"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

ROOT = r"D:\programme file\ResearchArticle"
OUT = os.path.join(ROOT, "results", "ppt_v3")
os.makedirs(OUT, exist_ok=True)

NAVY = "#0A1F44"; BLUE = "#1565C0"; ORANGE = "#E65100"
GREEN = "#2E7D32"; GRAY = "#555555"

# 横向布局：顶部描述 + 三列 Top 结果
fig = plt.figure(figsize=(12.5, 4.2), dpi=150)
fig.patch.set_facecolor("white")

# 顶部：标题 + 描述
fig.text(0.5, 0.88, "RAG 标准检索 · Top-3", fontsize=20, fontweight="bold",
         color=NAVY, ha="center")
fig.text(0.5, 0.78, "缺陷描述 → 匹配 GB/T 34528-2017 标准条款（TF-IDF + 余弦相似度）",
         fontsize=11, color=GRAY, ha="center")

# 缺陷描述条
box = FancyBboxPatch((0.02, 0.60), 0.96, 0.12, boxstyle="round,pad=0.01",
                     facecolor="#E3F2FD", edgecolor=BLUE, linewidth=1.8)
fig.patches.append(box)
fig.text(0.04, 0.635, "缺陷描述（VLM）：右上区域可见线状划痕，长约3-5mm，深度较浅，沿轴向延伸",
         fontsize=11, color=NAVY)

results = [
    ("Top-1", "GB34528-P07-S4", "第7页 · 划痕评定", "相似度 0.290",
     "划痕长度>5mm或深度>0.05mm影响表面质量，装饰面从严", GREEN),
    ("Top-2", "GB34528-P05-S2", "第5页 · 裂纹分类", "相似度 0.059",
     "裂纹是最危险缺陷之一，通常判定不合格", BLUE),
    ("Top-3", "GB34528-P10-S7", "第10页 · 检测条件", "相似度 0.044",
     "检测照度≥500lx，视线夹角≥30°", ORANGE),
]
for i, (rank, rid, sec, sim, text, c) in enumerate(results):
    x = 0.02 + i * 0.325
    box = FancyBboxPatch((x, 0.06), 0.31, 0.48, boxstyle="round,pad=0.01",
                         facecolor="white", edgecolor=c, linewidth=2)
    fig.patches.append(box)
    tag = FancyBboxPatch((x + 0.02, 0.43), 0.09, 0.08, boxstyle="round,pad=0.005",
                         facecolor=c, edgecolor="none")
    fig.patches.append(tag)
    fig.text(x + 0.065, 0.452, rank, fontsize=11, fontweight="bold", color="white", ha="center")
    fig.text(x + 0.03, 0.36, f"引用ID {rid}", fontsize=13, fontweight="bold", color=c)
    fig.text(x + 0.03, 0.29, sec, fontsize=10, color=GRAY)
    fig.text(x + 0.03, 0.225, sim, fontsize=10, color=c, fontweight="bold")
    fig.text(x + 0.03, 0.13, text, fontsize=9.5, color=GRAY)

plt.savefig(os.path.join(OUT, "retrieval_h.png"), bbox_inches="tight", facecolor="white")
plt.close()
print("retrieval_h.png 完成")
