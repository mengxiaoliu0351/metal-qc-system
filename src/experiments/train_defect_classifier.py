# -*- coding: utf-8 -*-
"""
#1 缺陷类型监督分类器
=====================
用 WinCLIP(ViT-B-16) 整图图像嵌入作为特征，在 MVTec metal_nut 带标签的
测试集上训一个 LogisticRegression 分类头，输出准确率与混淆矩阵。
与无监督异常检出(winclip_patch)互补：异常检出回答"有没有问题"，
本分类器回答"是什么类型的缺陷"。
"""
import os, sys, json, time
import numpy as np
from PIL import Image
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from perception.winclip_patch import WinCLIPPatch

# MVTec metal_nut 缺陷子目录 -> 中文标签
LABEL_MAP = {
    "good": "正常",
    "bent": "弯曲",
    "color": "色差",
    "flip": "翻面",
    "scratch": "划痕",
}
TEST_DIR = os.path.join(ROOT, "data", "mvtec", "metal_nut", "test")
OUT_PATH = os.path.join(ROOT, "results", "perception", "classifier_metal_nut.joblib")


def extract_features(det, files):
    import torch
    feats = []
    with torch.no_grad():
        for fp in files:
            img = Image.open(fp).convert("RGB")
            t = det.preprocess(img).unsqueeze(0).to(det.device)
            f = det.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f[0].cpu().numpy())
    return np.array(feats)


def main():
    import torch
    print("加载 CLIP 模型...")
    det = WinCLIPPatch("ViT-B-16")

    X, y = [], []
    classes = []
    for subdir, cn in LABEL_MAP.items():
        d = os.path.join(TEST_DIR, subdir)
        if not os.path.isdir(d):
            continue
        files = sorted(os.path.join(d, f) for f in os.listdir(d)
                       if f.lower().endswith((".png", ".jpg")))
        feats = extract_features(det, files)
        X.append(feats)
        y.extend([cn] * len(files))
        classes.append(cn)
        print(f"  {subdir:10s} -> {cn:4s}: {len(files)} 张")

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    print(f"\n总样本: {X.shape}, 特征维度: {X.shape[1]}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"训练集 {len(Xtr)} / 测试集 {len(Xte)}")

    # 5折交叉验证选正则强度C
    best_C, best_cv = 1.0, -1
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=3000, C=C))])
        cv = cross_val_score(pipe, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        print(f"  C={C:<6} 5折CV={cv*100:.1f}%")
        if cv > best_cv:
            best_cv, best_C = cv, C
    print(f"选用 C={best_C}")

    clf = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=3000, C=best_C))])
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"\n===== 测试集准确率: {acc*100:.2f}% =====")
    print("\n混淆矩阵 (行=真实, 列=预测):")
    labels = sorted(set(y))
    cm = confusion_matrix(yte, pred, labels=labels)
    print("        " + "  ".join(f"{l[:4]:>6s}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l[:6]:<6s}" + "  ".join(f"{cm[i,j]:6d}" for j in range(len(labels))))
    print("\n分类报告:")
    print(classification_report(yte, pred, labels=labels))

    # 保存
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    joblib.dump({"clf": clf, "labels": labels,
                 "label_map": LABEL_MAP, "acc_test": float(acc)}, OUT_PATH)
    print(f"\n已保存分类器 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
