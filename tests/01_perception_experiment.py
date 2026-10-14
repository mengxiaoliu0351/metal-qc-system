# -*- coding: utf-8 -*-
"""
感知层先导实验：WinCLIP 风格零样本异常检测
================================================
核心思路：
1. 用正常样本的 CLIP patch 特征构建记忆库
2. 测试图像每个 patch 与记忆库计算最大余弦相似度，相似度越低越异常
3. patch 级异常分数上采样得到热力图，整图取 max 作为异常分数
4. 用正常样本分数分布的 中位数 + 1.5*MAD 作为门控阈值
5. 选出高置信（明显缺陷）和低置信（边缘案例）样例

输出：
- results/perception/heatmap_*.png  缺陷热力图（3张典型缺陷）
- results/perception/score_distribution.png  正常/缺陷异常分数分布
- results/perception/high_conf_sample.png  高置信样例（标注分数）
- results/perception/low_conf_sample.png   低置信样例（标注分数）
- results/perception/scores.csv  所有测试图的异常分数
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import open_clip
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv

# 中文显示
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# ============ 路径配置 ============
ROOT = r"D:\programme file\ResearchArticle"
DATA_DIR = os.path.join(ROOT, "data", "mvtec", "metal_nut")
WEIGHTS_PATH = os.path.join(ROOT, "weights", "CLIP-ViT-B-16-open_clip_model.safetensors")
OUT_DIR = os.path.join(ROOT, "results", "perception")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# ============ 加载模型 ============
print("加载 CLIP ViT-B-16 模型...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained=WEIGHTS_PATH,
)
model = model.to(DEVICE).eval()
print("模型加载完成")

# ViT-B-16: patch_size=16, 224/16=14, 共196个patch
PATCH_SIZE = 16
IMG_SIZE = 224
N_PATCH = IMG_SIZE // PATCH_SIZE  # 14

def extract_patch_features(img_tensor):
    """提取图像的 patch 级特征 (196, 512)"""
    with torch.no_grad():
        # visual 内部: patch embedding -> positional embedding -> transformer
        # 我们直接调用 visual 的前几层获取 patch token
        x = model.visual.conv1(img_tensor)  # (B, 768, 14, 14) for ViT-B? 实际维度看模型
        # 用更稳妥的方式：获取中间输出
        # open_clip 的 VisionTransformer 有 forward 方法，我们可以 hook
        pass
    return None

# 用 hook 方式获取 patch token
patch_features = {}
def hook_fn(module, input, output):
    # output 是 transformer 的输出 (B, 197, 768)，包含 cls token
    patch_features["value"] = output[:, 1:, :]  # 去掉 cls token, (B, 196, dim)

# 注册 hook 到 visual transformer 的最后一层
handle = model.visual.transformer.resblocks[-1].register_forward_hook(hook_fn)

@torch.no_grad()
def get_patch_features(img_pil):
    """输入 PIL 图像，返回 (196, dim) 的 patch 特征（归一化）"""
    img_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)
    _ = model.encode_image(img_tensor)  # 触发 forward，hook 会捕获
    feats = patch_features["value"][0]  # (196, dim)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()

@torch.no_grad()
def build_memory_bank(normal_imgs, max_samples=100):
    """用正常样本构建 patch 特征记忆库
    返回: (N*196, dim) 所有正常样本所有 patch 的特征
    """
    print(f"构建正常样本特征记忆库（最多 {max_samples} 张）...")
    all_patch_feats = []
    for i, img_path in enumerate(normal_imgs[:max_samples]):
        img = Image.open(img_path).convert("RGB")
        feats = get_patch_features(img)
        all_patch_feats.append(feats)
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{min(len(normal_imgs), max_samples)} 张")
    memory = np.concatenate(all_patch_feats, axis=0)  # (N*196, dim)
    print(f"记忆库构建完成: {memory.shape[0]} 个 patch 特征")
    return memory

def compute_anomaly_map(img_pil, memory_bank):
    """计算图像的异常热力图 (14,14) 和整图异常分数"""
    feats = get_patch_features(img_pil)  # (196, dim)
    # 每个 patch 与记忆库中所有 patch 计算余弦相似度，取最大值
    # 相似度越低，异常程度越高
    sims = feats @ memory_bank.T  # (196, M)
    max_sims = sims.max(axis=1)  # (196,)
    # 异常分数 = 1 - 最大相似度
    anomaly_scores = 1.0 - max_sims  # (196,)
    anomaly_map = anomaly_scores.reshape(N_PATCH, N_PATCH)  # (14,14)
    # 整图异常分数 = patch 最大值
    img_score = float(anomaly_scores.max())
    return anomaly_map, img_score

def upsample_heatmap(anomaly_map, target_size=(224, 224)):
    """将 14x14 热力图上采样到目标大小"""
    from PIL import Image as PILImage
    hm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    hm_img = PILImage.fromarray((hm * 255).astype(np.uint8))
    hm_img = hm_img.resize(target_size, PILImage.BILINEAR)
    return np.array(hm_img) / 255.0

# ============ 主流程 ============
if __name__ == "__main__":
    # 1. 收集正常样本路径
    normal_dir = os.path.join(DATA_DIR, "train", "good")
    normal_paths = sorted([
        os.path.join(normal_dir, f) for f in os.listdir(normal_dir)
        if f.endswith(".png")
    ])
    print(f"正常样本: {len(normal_paths)} 张")

    # 2. 构建记忆库（用前80张正常样本，足够且快）
    memory_bank = build_memory_bank(normal_paths, max_samples=80)

    # 3. 收集测试样本
    test_dir = os.path.join(DATA_DIR, "test")
    test_samples = []  # (path, label)
    for label in os.listdir(test_dir):
        label_dir = os.path.join(test_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for f in sorted(os.listdir(label_dir)):
            if f.endswith(".png"):
                test_samples.append((os.path.join(label_dir, f), label))
    print(f"测试样本: {len(test_samples)} 张")

    # 4. 计算所有测试图的异常分数
    print("计算测试集异常分数...")
    results = []
    for i, (img_path, label) in enumerate(test_samples):
        img = Image.open(img_path).convert("RGB")
        anomaly_map, score = compute_anomaly_map(img, memory_bank)
        results.append({
            "path": img_path,
            "label": label,
            "score": score,
            "anomaly_map": anomaly_map,
        })
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{len(test_samples)} 张")

    # 5. 计算正常测试样本的分数分布，确定阈值
    normal_test_scores = [r["score"] for r in results if r["label"] == "good"]
    defect_scores = [r["score"] for r in results if r["label"] != "good"]
    print(f"\n正常测试样本分数: mean={np.mean(normal_test_scores):.4f}, "
          f"median={np.median(normal_test_scores):.4f}, "
          f"std={np.std(normal_test_scores):.4f}")
    print(f"缺陷测试样本分数: mean={np.mean(defect_scores):.4f}, "
          f"median={np.median(defect_scores):.4f}, "
          f"std={np.std(defect_scores):.4f}")

    # 阈值 = 中位数 + 1.5 * MAD（中位数绝对偏差）
    normal_median = np.median(normal_test_scores)
    mad = np.median(np.abs(normal_test_scores - normal_median))
    threshold = normal_median + 1.5 * mad
    print(f"\n阈值计算: 中位数={normal_median:.4f}, MAD={mad:.4f}, "
          f"阈值(中位数+1.5*MAD)={threshold:.4f}")

    # 6. 保存分数 CSV
    csv_path = os.path.join(OUT_DIR, "scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["文件名", "类别", "异常分数", "是否高于阈值"])
        for r in results:
            fname = os.path.basename(r["path"])
            writer.writerow([fname, r["label"], f"{r['score']:.4f}",
                             "是" if r["score"] >= threshold else "否"])
    print(f"分数已保存: {csv_path}")

    # 7. 绘制分数分布图
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(min(normal_test_scores), min(defect_scores)),
        max(max(normal_test_scores), max(defect_scores)),
        20,
    )
    ax.hist(normal_test_scores, bins=bins, alpha=0.6, label="正常样本", color="steelblue")
    ax.hist(defect_scores, bins=bins, alpha=0.6, label="缺陷样本", color="coral")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label=f"门控阈值 = {threshold:.4f}")
    ax.set_xlabel("异常分数", fontsize=12)
    ax.set_ylabel("样本数量", fontsize=12)
    ax.set_title("正常/缺陷样本异常分数分布（MVTec metal_nut）", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    dist_path = os.path.join(OUT_DIR, "score_distribution.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"分数分布图已保存: {dist_path}")

    # 8. 保存3张典型缺陷热力图
    defect_results = [r for r in results if r["label"] != "good"]
    # 按分数排序，选高、中、低各一张
    defect_results_sorted = sorted(defect_results, key=lambda x: x["score"], reverse=True)
    picks = [
        defect_results_sorted[0],  # 最高
        defect_results_sorted[len(defect_results_sorted)//2],  # 中间
        defect_results_sorted[-1],  # 最低
    ]
    for idx, r in enumerate(picks):
        img = Image.open(r["path"]).convert("RGB").resize((224, 224))
        hm = upsample_heatmap(r["anomaly_map"])
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img)
        axes[0].set_title(f"原图 ({r['label']})", fontsize=12)
        axes[0].axis("off")
        im = axes[1].imshow(hm, cmap="jet", vmin=0, vmax=1)
        axes[1].imshow(img, alpha=0.3)
        axes[1].set_title(f"异常热力图 (分数={r['score']:.4f})", fontsize=12)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        hm_path = os.path.join(OUT_DIR, f"heatmap_{idx+1}_{r['label']}.png")
        plt.savefig(hm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"热力图 {idx+1} 已保存: {hm_path}")

    # 9. 高置信样例（缺陷，分数远高于阈值）
    high_conf = [r for r in defect_results if r["score"] >= threshold * 1.3]
    if high_conf:
        # 选分数最高的
        r = max(high_conf, key=lambda x: x["score"])
    else:
        r = defect_results_sorted[0]
    img = Image.open(r["path"]).convert("RGB").resize((224, 224))
    hm = upsample_heatmap(r["anomaly_map"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(img)
    axes[0].set_title(f"原图 - {r['label']}", fontsize=13)
    axes[0].axis("off")
    im = axes[1].imshow(hm, cmap="jet", vmin=0, vmax=1)
    axes[1].imshow(img, alpha=0.3)
    axes[1].set_title(f"异常热力图", fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"【高置信样本】走快链路 | 异常分数: {r['score']:.4f} | 阈值: {threshold:.4f} | "
        f"分数/阈值 = {r['score']/threshold:.2f}x",
        fontsize=13, color="darkred", fontweight="bold",
    )
    plt.tight_layout()
    high_path = os.path.join(OUT_DIR, "high_conf_sample.png")
    plt.savefig(high_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"高置信样例已保存: {high_path}")

    # 10. 低置信样例（分数接近阈值的边缘案例）
    # 找分数在阈值 0.8~1.2 倍之间的缺陷样本
    low_conf = [r for r in defect_results
                if threshold * 0.7 <= r["score"] <= threshold * 1.2]
    if low_conf:
        # 选最接近阈值的
        r = min(low_conf, key=lambda x: abs(x["score"] - threshold))
    else:
        # 找分数最低的缺陷样本
        r = defect_results_sorted[-1]
    img = Image.open(r["path"]).convert("RGB").resize((224, 224))
    hm = upsample_heatmap(r["anomaly_map"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(img)
    axes[0].set_title(f"原图 - {r['label']}", fontsize=13)
    axes[0].axis("off")
    im = axes[1].imshow(hm, cmap="jet", vmin=0, vmax=1)
    axes[1].imshow(img, alpha=0.3)
    axes[1].set_title(f"异常热力图", fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"【低置信样本】触发慢链路深度分析 | 异常分数: {r['score']:.4f} | "
        f"阈值: {threshold:.4f} | 分数/阈值 = {r['score']/threshold:.2f}x",
        fontsize=13, color="darkorange", fontweight="bold",
    )
    plt.tight_layout()
    low_path = os.path.join(OUT_DIR, "low_conf_sample.png")
    plt.savefig(low_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"低置信样例已保存: {low_path}")

    # 11. 统计检测效果（粗略 AUROC）
    from sklearn.metrics import roc_auc_score
    y_true = [1 if r["label"] != "good" else 0 for r in results]
    y_scores = [r["score"] for r in results]
    auroc = roc_auc_score(y_true, y_scores)
    print(f"\n=== 检测效果（零样本，无训练） ===")
    print(f"AUROC: {auroc:.4f}")
    # 阈值下的准确率
    y_pred = [1 if s >= threshold else 0 for s in y_scores]
    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    print(f"阈值={threshold:.4f} 下: 准确率={acc:.4f}, "
          f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    if tp + fp > 0:
        print(f"精确率={tp/(tp+fp):.4f}, 召回率={tp/(tp+fn):.4f}")

    # 保存指标
    metrics_path = os.path.join(OUT_DIR, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"数据集: MVTec-AD metal_nut\n")
        f.write(f"模型: CLIP ViT-B-16 (open_clip)\n")
        f.write(f"正常记忆库样本数: 80\n")
        f.write(f"测试样本数: {len(results)}\n")
        f.write(f"阈值(中位数+1.5*MAD): {threshold:.4f}\n")
        f.write(f"AUROC: {auroc:.4f}\n")
        f.write(f"准确率: {acc:.4f}\n")
        f.write(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}\n")
    print(f"指标已保存: {metrics_path}")

    handle.remove()  # 移除 hook
    print("\n=== 感知层先导实验全部完成 ===")
