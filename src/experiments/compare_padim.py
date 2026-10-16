# -*- coding: utf-8 -*-
"""
方法对比实验：最近邻余弦距离（现有） vs PaDiM 马氏距离（自研）
在 metal_nut 完整测试集上算 AUC，证明"自己实现的算法"带来的提升。
"""
import os, glob, numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, r"D:\programme file\ResearchArticle\src")
from perception.winclip_patch import WinCLIPPatch, N_PATCH

ROOT = r"D:\programme file\ResearchArticle"
TEST = os.path.join(ROOT, "data", "mvtec", "metal_nut", "test")
BANK = os.path.join(ROOT, "results", "perception", "banks", "金属螺母(metal_nut).npy")

print("加载模型...")
det = WinCLIPPatch("ViT-B-16")
bank = np.load(BANK).astype(np.float64)          # [15680, 512]
print("bank:", bank.shape)

# ===== 现有方法：最近邻余弦距离（detect 里的逻辑）=====
def score_nn(feats):
    sims = feats @ bank.T
    max_sims = sims.max(axis=1)
    return float((1.0 - max_sims).max())

# ===== PaDiM 马氏距离（自研算法）=====
mu = bank.mean(axis=0)
Sigma = np.cov(bank, rowvar=False)
# 正则化：小样本协方差求逆会震荡，加 ridge
reg = 0.01 * np.trace(Sigma) / Sigma.shape[0]
Sigma_inv = np.linalg.inv(Sigma + reg * np.eye(Sigma.shape[0]))
print("PaDiM 协方差已估计，正则化项=", round(reg, 5))

def score_mahalanobis(feats):
    d = feats - mu
    # 每个patch的马氏距离平方
    m = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', d, Sigma_inv, d), 0))
    return float(m.max())

# ===== 遍历测试集 =====
rows = []   # (score_nn, score_maha, label)
for label, name in [(0, "good"), (1, "bent"), (1, "color"), (1, "flip"), (1, "scratch")]:
    files = sorted(glob.glob(os.path.join(TEST, name, "*.png")))
    print(f"  {name:<10} {len(files)} 张")
    for f in files:
        img = Image.open(f).convert("RGB")
        feats = det._get_patch_feats(img).astype(np.float64)
        rows.append((score_nn(feats), score_mahalanobis(feats), label))

rows = np.array(rows)
y_true = rows[:, 2].astype(int)
auc_nn = roc_auc_score(y_true, rows[:, 0])
auc_maha = roc_auc_score(y_true, rows[:, 1])

print("\n===== AUC 对比（metal_nut 全测试集 {} 张）=====".format(len(rows)))
print(f"  最近邻余弦距离（现有） : {auc_nn*100:.2f}%")
print(f"  PaDiM 马氏距离（自研） : {auc_maha*100:.2f}%")
print(f"  差异: {(auc_maha-auc_nn)*100:+.2f} 个百分点")
