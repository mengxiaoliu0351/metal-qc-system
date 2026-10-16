# -*- coding: utf-8 -*-
"""
#1 监督缺陷类型分类器封装
=========================
加载训练好的 LogisticRegression(在CLIP整图特征上)，对传入图片给出
缺陷类型预测与置信度。与 winclip_patch 的零样本文本分类互补：
零样本靠"文本提示"猜类型，本分类器靠"带标签样本"学类型。
"""
import os
import numpy as np
from PIL import Image
import joblib

ROOT = r"D:\programme file\ResearchArticle"
CLF_PATH = os.path.join(ROOT, "results", "perception", "classifier_metal_nut.joblib")


class DefectClassifier:
    def __init__(self, clf_path=CLF_PATH):
        self.bundle = None
        self.labels = None
        if clf_path and os.path.exists(clf_path):
            self.bundle = joblib.load(clf_path)
            self.labels = self.bundle.get("labels")

    @property
    def available(self):
        return self.bundle is not None

    def predict(self, det, image):
        """det: 已加载的 WinCLIPPatch；image: 路径或PIL。
        返回 {"label":str, "confidence":float, "probs":{label:prob}}"""
        if not self.available:
            return None
        import torch
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        t = det.preprocess(image).unsqueeze(0).to(det.device)
        with torch.no_grad():
            f = det.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        feat = f[0].cpu().numpy().reshape(1, -1)
        clf = self.bundle["clf"]
        label = clf.predict(feat)[0]
        proba = clf.predict_proba(feat)[0]
        probs = {l: float(p) for l, p in zip(clf.classes_, proba)}
        return {"label": label, "confidence": float(max(proba)), "probs": probs}


if __name__ == "__main__":
    det = None
    print("分类器封装自检：模块可独立加载，predict 需传入 WinCLIPPatch 实例")
