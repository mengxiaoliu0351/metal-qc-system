# -*- coding: utf-8 -*-
"""
阈值train/test分离：在train/good上算阈值（中位数+1.5×MAD），再在test集上测AUROC
避免数据泄露。
"""
import os, sys, json
import numpy as np
from PIL import Image

ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT,"src"))
from perception.winclip_patch import WinCLIPPatch

det = WinCLIPPatch("ViT-B-16")
det.build_bank(os.path.join(ROOT,"data","mvtec","metal_nut","train","good"),
               max_samples=80,
               cache_path=os.path.join(ROOT,"results","perception","memory_bank_80.npy"))

# 1) 在train/good上算每张图的整图分数（不是patch，是整图max）
print("=== 在train/good(80张)上算分数分布 ===")
train_dir = os.path.join(ROOT,"data","mvtec","metal_nut","train","good")
train_scores = []
for fn in sorted(os.listdir(train_dir)):
    if not fn.endswith(".png"): continue
    img = Image.open(os.path.join(train_dir,fn)).convert("RGB")
    r = det.detect(img)
    train_scores.append(float(r["score"]))
train_scores = np.array(train_scores)
median = np.median(train_scores)
mad = np.median(np.abs(train_scores - median))
new_thr = median + 1.5 * mad
print(f"train分数: mean={train_scores.mean():.4f} median={median:.4f} MAD={mad:.4f}")
print(f"新阈值 = median + 1.5*MAD = {new_thr:.4f}")

# 2) 在test集上测
print("\n=== 在test集(115张)上测 ===")
test_dir = os.path.join(ROOT,"data","mvtec","metal_nut","test")
records = []
for cat in sorted(os.listdir(test_dir)):
    cat_dir = os.path.join(test_dir,cat)
    if not os.path.isdir(cat_dir): continue
    for fn in sorted(os.listdir(cat_dir)):
        if not fn.endswith(".png"): continue
        img = Image.open(os.path.join(cat_dir,fn)).convert("RGB")
        r = det.detect(img)
        records.append({"cat":cat,"file":fn,
                        "label":0 if cat=="good" else 1,
                        "score":float(r["score"])})

scores = np.array([r["score"] for r in records])
labels = np.array([r["label"] for r in records])
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
auroc = roc_auc_score(labels, scores)
preds = (scores >= new_thr).astype(int)
acc = accuracy_score(labels, preds)
cm = confusion_matrix(labels, preds)
print(f"\nAUROC: {auroc:.4f}")
print(f"准确率(新阈值{new_thr:.4f}): {acc:.4f}")
print(f"混淆矩阵 TN FP FN TP = {cm.tolist()}")

# 保存
out = {"new_threshold": round(float(new_thr),4),
       "train_median": round(float(median),4), "train_mad": round(float(mad),4),
       "auroc": round(float(auroc),4), "accuracy": round(float(acc),4),
       "confusion_matrix": cm.tolist(),
       "n_test": len(records)}
with open(os.path.join(ROOT,"results","experiments","exp1_train_test_split.json"),"w",encoding="utf-8") as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
print("\n已保存 exp1_train_test_split.json")
