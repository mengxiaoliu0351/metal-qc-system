# -*- coding: utf-8 -*-
"""
生成 PPT v3 专用精简版实验图
设计原则：大字、少文字、强对比、突出关键数字
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

ROOT = r"D:\programme file\ResearchArticle"
OUT = os.path.join(ROOT, "results", "ppt_v3")
os.makedirs(OUT, exist_ok=True)
PER = os.path.join(ROOT, "results", "perception")

# 配色
NAVY = "#0A1F44"; BLUE = "#1565C0"; BLUE_L = "#42A5F5"
ORANGE = "#E65100"; ORANGE_BG = "#FFF3E0"
GREEN = "#2E7D32"; GREEN_BG = "#E8F5E9"
RED = "#C62828"; RED_BG = "#FFEBEE"
PURPLE = "#6A1B9A"; PURPLE_BG = "#F3E5F5"
GRAY = "#555555"; GRAY_L = "#9E9E9E"; BG = "#FAFAFA"

# ============================================================
# 图1：双链路门控流程（精简版） 16:9 高清
# ============================================================
def make_dual_path():
    fig = plt.figure(figsize=(13.33, 7.5), dpi=150)
    fig.patch.set_facecolor("white")

    # 顶部标题
    fig.text(0.5, 0.94, "双链路门控机制", fontsize=26, fontweight="bold",
             color=NAVY, ha="center")
    fig.text(0.5, 0.88, "高置信走快链路 · 低置信触发慢链路深度分析",
             fontsize=14, color=GRAY, ha="center")

    # 输入 + 门控
    box = FancyBboxPatch((0.03, 0.60), 0.14, 0.18, boxstyle="round,pad=0.01",
                         facecolor="#E3F2FD", edgecolor=BLUE, linewidth=2)
    fig.patches.append(box)
    fig.text(0.10, 0.70, "输入图像", fontsize=13, fontweight="bold", ha="center", color=NAVY)
    fig.text(0.10, 0.635, "WinCLIP\n异常分数", fontsize=10, ha="center", color=GRAY)

    gate = FancyBboxPatch((0.20, 0.58), 0.16, 0.22, boxstyle="round,pad=0.01",
                          facecolor=ORANGE_BG, edgecolor=ORANGE, linewidth=2.5)
    fig.patches.append(gate)
    fig.text(0.28, 0.735, "置信度门控", fontsize=13, fontweight="bold", ha="center", color=ORANGE)
    fig.text(0.28, 0.66, "阈值 = 0.2243", fontsize=10, ha="center", color=GRAY)
    fig.text(0.28, 0.615, "≥阈值→快  <阈值→慢", fontsize=9, ha="center", color=GRAY)

    fig.patches.append(FancyArrowPatch((0.17, 0.69), (0.20, 0.69),
                       arrowstyle="->", color=BLUE, linewidth=2, mutation_scale=15))

    # 左：快链路
    box = FancyBboxPatch((0.40, 0.60), 0.28, 0.20, boxstyle="round,pad=0.01",
                         facecolor=GREEN_BG, edgecolor=GREEN, linewidth=2)
    fig.patches.append(box)
    fig.text(0.46, 0.755, "快链路（高置信 ≥ 0.2243）", fontsize=14, fontweight="bold",
             color=GREEN, ha="left")
    fig.text(0.46, 0.68, "异常热力图 → 缺陷定位 → 等级判定", fontsize=12, ha="left", color=GRAY)
    fig.text(0.46, 0.63, "耗时 2-3 秒 · 省算力", fontsize=12, fontweight="bold", color=ORANGE)

    # 右：慢链路
    box = FancyBboxPatch((0.72, 0.60), 0.25, 0.20, boxstyle="round,pad=0.01",
                         facecolor=PURPLE_BG, edgecolor=PURPLE, linewidth=2)
    fig.patches.append(box)
    fig.text(0.76, 0.755, "慢链路（低置信 < 0.2243）", fontsize=14, fontweight="bold",
             color=PURPLE, ha="left")
    fig.text(0.76, 0.68, "裁剪 → VLM描述 → RAG检索 → 报告", fontsize=11, ha="left", color=GRAY)
    fig.text(0.76, 0.63, "耗时 20-30 秒 · 深度分析", fontsize=12, fontweight="bold", color=ORANGE)

    # 箭头
    fig.patches.append(FancyArrowPatch((0.36, 0.69), (0.40, 0.69),
                       arrowstyle="->", color=GREEN, linewidth=2.5, mutation_scale=15))
    fig.patches.append(FancyArrowPatch((0.68, 0.69), (0.72, 0.69),
                       arrowstyle="->", color=PURPLE, linewidth=2.5, mutation_scale=15))

    # 底部：实际结果对比（用感知层样例图）
    try:
        high = Image.open(os.path.join(PER, "high_conf_sample.png"))
        low = Image.open(os.path.join(PER, "low_conf_sample.png"))
        # 等比例放入
        ax1 = fig.add_axes([0.04, 0.05, 0.45, 0.50])
        ax1.imshow(high); ax1.axis("off")
        ax1.set_title("高置信样例 → 快链路直接输出", fontsize=13, fontweight="bold", color=GREEN, pad=8)
        ax2 = fig.add_axes([0.51, 0.05, 0.45, 0.50])
        ax2.imshow(low); ax2.axis("off")
        ax2.set_title("低置信样例 → 慢链路深度分析", fontsize=13, fontweight="bold", color=PURPLE, pad=8)
    except Exception as e:
        fig.text(0.25, 0.3, "[样例图]", fontsize=14, ha="center")
        fig.text(0.75, 0.3, "[样例图]", fontsize=14, ha="center")

    plt.savefig(os.path.join(OUT, "dual_path_v3.png"), bbox_inches="tight", facecolor="white")
    plt.close()
    print("dual_path_v3.png 完成")

# ============================================================
# 图2：RAG 检索（精简版）16:9
# ============================================================
def make_retrieval():
    fig = plt.figure(figsize=(13.33, 7.5), dpi=150)
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.93, "RAG 标准检索 · Top-3", fontsize=24, fontweight="bold",
             color=NAVY, ha="center")
    fig.text(0.5, 0.87, "缺陷描述 → 匹配 GB/T 34528-2017 标准条款", fontsize=13,
             color=GRAY, ha="center")

    # 缺陷描述
    box = FancyBboxPatch((0.04, 0.72), 0.92, 0.10, boxstyle="round,pad=0.01",
                         facecolor="#E3F2FD", edgecolor=BLUE, linewidth=2)
    fig.patches.append(box)
    fig.text(0.06, 0.75, "缺陷描述（VLM）：在零件右上区域可见一条线状划痕，长度约3-5mm，深度较浅，沿轴向延伸。",
             fontsize=12, color=NAVY)

    results = [
        ("Top-1", "GB34528-P07-S4", "第7页 · 划痕评定", "0.290", "划痕长度>5mm或深度>0.05mm影响表面质量，装饰面从严", GREEN),
        ("Top-2", "GB34528-P05-S2", "第5页 · 裂纹分类", "0.059", "裂纹是最危险缺陷之一，通常判定不合格", BLUE),
        ("Top-3", "GB34528-P10-S7", "第10页 · 检测条件", "0.044", "检测照度≥500lx，视线夹角≥30°", ORANGE),
    ]
    for i, (rank, rid, sec, sim, text, c) in enumerate(results):
        y = 0.52 - i * 0.20
        box = FancyBboxPatch((0.04, y), 0.92, 0.17, boxstyle="round,pad=0.01",
                             facecolor="white", edgecolor=c, linewidth=2)
        fig.patches.append(box)
        # 排名标签
        tag = FancyBboxPatch((0.06, y + 0.04), 0.08, 0.09, boxstyle="round,pad=0.005",
                             facecolor=c, edgecolor="none")
        fig.patches.append(tag)
        fig.text(0.10, y + 0.065, rank, fontsize=12, fontweight="bold", color="white", ha="center")
        # ID + 相似度
        fig.text(0.17, y + 0.085, f"引用ID: {rid}", fontsize=15, fontweight="bold", color=c)
        fig.text(0.17, y + 0.03, f"{sec}    |    相似度 {sim}", fontsize=11, color=GRAY)
        fig.text(0.60, y + 0.065, text, fontsize=10, color=GRAY)

    fig.text(0.5, 0.035, "每条标准片段绑定 引用ID + 页码 + 标准号，可逐条溯源",
             fontsize=13, color=GRAY, ha="center")

    plt.savefig(os.path.join(OUT, "retrieval_v3.png"), bbox_inches="tight", facecolor="white")
    plt.close()
    print("retrieval_v3.png 完成")

# ============================================================
# 图3：引用校验通过 vs 失败（左右对比）16:9
# ============================================================
def make_validation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.33, 5.0), dpi=150)
    fig.patch.set_facecolor("white")

    # 左：通过
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis("off")
    box = FancyBboxPatch((0.3, 1.0), 9.4, 8.2, boxstyle="round,pad=0.2",
                         facecolor=GREEN_BG, edgecolor=GREEN, linewidth=3)
    ax1.add_patch(box)
    ax1.text(5, 8.6, "【校验通过】", fontsize=20, fontweight="bold", color=GREEN, ha="center")
    ax1.text(1, 6.8, "VLM 输出引用 ID：", fontsize=12, color=GRAY)
    ax1.text(1, 5.9, "GB34528-P07-S4\nGB34528-P10-S7", fontsize=13, fontweight="bold", color=NAVY,
            family="Consolas")
    ax1.text(1, 4.2, "检索结果中包含：", fontsize=12, color=GRAY)
    ax1.text(1, 3.3, "[√] GB34528-P07-S4\n[√] GB34528-P10-S7", fontsize=13, color=GREEN,
            family="Consolas")
    ax1.text(5, 2.0, "所有引用均在检索结果中\n正常标注标准号 + 页码展示",
             fontsize=13, fontweight="bold", color=GREEN, ha="center")

    # 右：失败
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
    box = FancyBboxPatch((0.3, 1.0), 9.4, 8.2, boxstyle="round,pad=0.2",
                         facecolor=RED_BG, edgecolor=RED, linewidth=3)
    ax2.add_patch(box)
    ax2.text(5, 8.6, "【编造引用 → 降级】", fontsize=20, fontweight="bold", color=RED, ha="center")
    ax2.text(1, 6.8, "VLM 输出引用 ID：", fontsize=12, color=GRAY)
    ax2.text(1, 5.9, "GB34528-P09-S6\nGB34528-P99-S99", fontsize=13, fontweight="bold", color=NAVY,
            family="Consolas")
    ax2.text(1, 4.2, "其中：", fontsize=12, color=GRAY)
    ax2.text(1, 3.3, "GB34528-P99-S99", fontsize=13, fontweight="bold", color=RED,
            family="Consolas")
    ax2.text(3.7, 3.3, " 不存在！", fontsize=13, fontweight="bold", color=RED)
    ax2.text(5, 2.0, "自动标红「引用待核验」\n降级为展示原始标准片段 · 人工选择",
             fontsize=13, fontweight="bold", color=RED, ha="center")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "validation_v3.png"), bbox_inches="tight", facecolor="white")
    plt.close()
    print("validation_v3.png 完成")

# ============================================================
# 图4：认知层四步流程（精简版）
# ============================================================
def make_cognition():
    fig, axes = plt.subplots(1, 4, figsize=(13.33, 3.6), dpi=150)
    fig.patch.set_facecolor("white")
    steps = [
        ("① VLM 描述", "裁剪缺陷图\n生成自然语言描述\n形态/位置/程度", BLUE, "#E3F2FD"),
        ("② RAG 检索", "匹配标准条款\n返回引用ID+页码\nTop-3 候选", ORANGE, ORANGE_BG),
        ("③ 引用校验", "检查引用ID\n是否在检索结果中\n错误可识别", GREEN, GREEN_BG),
        ("④ 结构化输出", "结论+缺陷描述\n引用ID+处置建议\n可溯源报告", PURPLE, PURPLE_BG),
    ]
    for ax, (title, content, c, bg) in zip(axes, steps):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        box = FancyBboxPatch((0.4, 0.8), 9.2, 8.4, boxstyle="round,pad=0.2",
                             facecolor=bg, edgecolor=c, linewidth=2.5)
        ax.add_patch(box)
        ax.text(5, 8.0, title, fontsize=16, fontweight="bold", color=c, ha="center")
        ax.text(5, 5.5, content, fontsize=12, color=NAVY, ha="center", va="center",
                linespacing=1.6)
    # 箭头
    for i in range(3):
        axes[i].annotate("", xy=(10.6, 5), xytext=(10.2, 5),
                         arrowprops=dict(arrowstyle="->", color=GRAY, lw=2.5))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cognition_v3.png"), bbox_inches="tight", facecolor="white")
    plt.close()
    print("cognition_v3.png 完成")

# ============================================================
# 图5：风险预案六宫格（精简版）
# ============================================================
def make_risks():
    fig = plt.figure(figsize=(13.33, 7.5), dpi=150)
    fig.patch.set_facecolor("white")
    risks = [
        ("WinCLIP 效果不达预期", "→ 替换 PatchCore / Anomalib", RED, "#FFEBEE"),
        ("VLM 速度过慢", "→ 轻量化模型 / 模板化生成", ORANGE, "#FFF3E0"),
        ("RAG 匹配不准", "→ 优化分块 / 人工预置条目", BLUE, "#E3F2FD"),
        ("门控阈值不合理", "→ MAD辅助标定 + 人工微调", GREEN, "#E8F5E9"),
        ("VLM 不服从 JSON", "→ 正则提取 / 降级人工选择", PURPLE, "#F3E5F5"),
        ("真实样本难获取", "→ 公开工业数据集跨域验证", GRAY, "#EEEEEE"),
    ]
    for i, (title, sol, c, bg) in enumerate(risks):
        row, col = i // 3, i % 3
        x = 0.02 + col * 0.33
        y = 0.52 - row * 0.42
        box = FancyBboxPatch((x, y), 0.31, 0.38, boxstyle="round,pad=0.01",
                             facecolor=bg, edgecolor=c, linewidth=2)
        fig.patches.append(box)
        fig.text(x + 0.02, y + 0.30, title, fontsize=14, fontweight="bold", color=c)
        fig.text(x + 0.02, y + 0.12, sol, fontsize=12, color=NAVY)
    fig.text(0.5, 0.05, "核心原则：系统永远可用 — 任何模块失效都有降级方案",
             fontsize=14, fontweight="bold", color=ORANGE, ha="center")
    plt.savefig(os.path.join(OUT, "risks_v3.png"), bbox_inches="tight", facecolor="white")
    plt.close()
    print("risks_v3.png 完成")

if __name__ == "__main__":
    make_dual_path()
    make_retrieval()
    make_validation()
    make_cognition()
    make_risks()
    print("全部精简版实验图生成完成:", OUT)
