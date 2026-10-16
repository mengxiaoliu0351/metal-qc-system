# -*- coding: utf-8 -*-
"""
多距离度量对比（同一批特征，避免重复提特征）：
  1NN 余弦（现有）/ kNN余弦(k=5) / 按空间位置PaDiM马氏 / 特征白化欧氏
目标：找到能超过 96.58% 的自研改进。
"""
import os, glob, numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
import sys
sys.path.insert(0, r"D:\programme file\ResearchArticle\src")
from perception.winclip_patch import WinCLIPPatch

ROOT = r"D:\programme file\ResearchArticle"
TEST = os.path.join(ROOT, "data", "mvtec", "metal_nut", "test")
TRAIN_GOOD = os.path.join(ROOT, "data", "mvtec", "metal_nut", "train", "good")

print("加载模型...")
det = WinCLIPPatch("ViT-B-16")

# ===== 按空间位置的正态bank：[N_img, 196, D] =====
files = sorted(f for f in os.listdir(TRAIN_GOOD) if f.endswith(".png"))[:80]
pos_bank = []
for f in files:
    img = Image.open(os.path.join(TRAIN_GOOD, f)).convert("RGB")
    pos_bank.append(det._get_patch_feats(img))
pos_bank = np.stack(pos_bank, 0).astype(np.float64)   # [80,196,768]
flat_bank = pos_bank.reshape(-1, pos_bank.shape[-1])   # [15680,768]
print("pos_bank:", pos_bank.shape)

# 标准PaDiM：每个空间位置的协方差（80样本估768维，用LedoitWolf收缩+降维）
# 用PCA降到50维再估协方差
from sklearn.decomposition import PCA
pca = PCA(n_components=50, random_state=0).fit(flat_bank)
pb_pca = pca.transform(pos_bank.reshape(-1, pos_bank.shape[-1])).reshape(pos_bank.shape[0], 196, 50)
mu_pos = pb_pca.mean(0)                                   # [196,50]
sw = LedoitWolf().fit(pb_pca.reshape(-1,50))
cov_inv_pos = np.linalg.inv(sw.covariance_ + 1e-4*np.eye(50))  # [50,50] 共享

# ===== 提所有测试图特征 =====
X, y = [], []
for label, name in [(0,"good"),(1,"bent"),(1,"color"),(1,"flip"),(1,"scratch")]:
    for f in sorted(glob.glob(os.path.join(TEST,name,"*.png"))):
        img = Image.open(f).convert("RGB")
        X.append(det._get_patch_feats(img).astype(np.float64))
        y.append(label)
y = np.array(y)
print("测试图:", len(y))

def score_1nn(feats):
    sims = feats @ flat_bank.T
    return float((1.0 - sims.max(1)).max())

def score_knn(feats, k=5):
    sims = feats @ flat_bank.T
    topk = np.sort(sims, axis=1)[:, -k:].mean(1)   # top-k平均相似度
    return float((1.0 - topk).max())

def score_padim_pos(feats):
    f_pca = pca.transform(feats)                    # [196,50]
    d = f_pca - mu_pos                             # [196,50]
    m = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', d, cov_inv_pos, d),0))
    return float(m.max())

results = {}
for name, fn in [("1NN余弦(现有)", score_1nn), ("kNN-5余弦", score_knn),
                 ("PaDiM位置马氏(降维)", score_padim_pos)]:
    s = np.array([fn(x) for x in X])
    results[name] = roc_auc_score(y, s)

print("\n===== AUC 对比 (metal_nut 115张) =====")
for k,v in results.items():
    print(f"  {k:<22}: {v*100:.2f}%")
