# -*- coding: utf-8 -*-
"""通用缺陷分类头训练：参数化产品（screw / grid / metal_nut）"""
import os, sys, argparse
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

LABEL_MAPS = {
    "screw": {
        "good": "正常", "manipulated_front": "前端变形",
        "scratch_head": "头部划痕", "scratch_neck": "颈部划痕",
        "thread_side": "侧面螺纹", "thread_top": "顶部螺纹",
    },
    "grid": {
        "good": "正常", "bent": "弯曲", "broken": "破损",
        "glue": "胶痕", "metal_contamination": "金属污染", "thread": "划痕",
    },
}


def extract(det, files):
    import torch
    out = []
    with torch.no_grad():
        for fp in files:
            img = Image.open(fp).convert("RGB")
            t = det.preprocess(img).unsqueeze(0).to(det.device)
            f = det.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f[0].cpu().numpy())
    return np.array(out)


def main(product):
    lmap = LABEL_MAPS[product]
    test_dir = os.path.join(ROOT, "data", "mvtec", product, "test")
    out_path = os.path.join(ROOT, "results", "perception", f"classifier_{product}.joblib")
    import torch
    det = WinCLIPPatch("ViT-B-16")
    X, y = [], []
    for sub, cn in lmap.items():
        d = os.path.join(test_dir, sub)
        if not os.path.isdir(d):
            continue
        files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith((".png",".jpg"))]
        X.append(extract(det, files)); y.extend([cn]*len(files))
        print(f"  {sub:22s} -> {cn:6s}: {len(files)}")
    X = np.concatenate(X, 0); y = np.array(y)
    print(f"总样本 {X.shape}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    best_C, best_cv = 1.0, -1
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=3000, C=C))])
        cv = cross_val_score(pipe, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        if cv > best_cv:
            best_cv, best_C = cv, C
    clf = Pipeline([("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=3000, C=best_C))])
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    labels = sorted(set(y))
    cm = confusion_matrix(yte, pred, labels=labels)
    print(f"\n===== {product} 测试集准确率: {acc*100:.2f}% (C={best_C}, CV={best_cv*100:.1f}%) =====")
    print("        " + "  ".join(f"{l[:4]:>6s}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l[:6]:<6s}" + "  ".join(f"{cm[i,j]:6d}" for j in range(len(labels))))
    print("\n分类报告 (含 Macro-F1):")
    print(classification_report(yte, pred, labels=labels))
    joblib.dump({"clf": clf, "labels": labels, "label_map": lmap, "acc_test": float(acc)}, out_path)
    print(f"已保存 -> {out_path}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("product", choices=list(LABEL_MAPS))
    a = ap.parse_args()
    main(a.product)
