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
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLF_PATH = os.path.join(ROOT, "results", "perception", "classifier_metal_nut.joblib")


class DefectClassifier:
    def __init__(self, clf_path=CLF_PATH):
        self.bundle = None
        self.labels = None
        self.params = None
        if clf_path and os.path.exists(clf_path):
            # Read numerical parameters rather than calling a pickled sklearn
            # estimator across versions (the old files lack multi_class).
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")
                self.bundle = joblib.load(clf_path)
            pipeline = self.bundle["clf"]
            scaler, classifier = pipeline.named_steps["sc"], pipeline.named_steps["clf"]
            self.params = {
                "mean": np.asarray(scaler.mean_), "scale": np.asarray(scaler.scale_),
                "coef": np.asarray(classifier.coef_),
                "intercept": np.asarray(classifier.intercept_),
                "classes": np.asarray(classifier.classes_),
                "ovr": (getattr(classifier, "multi_class", None) == "ovr"
                        or getattr(classifier, "solver", None) == "liblinear"),
            }
            self.labels = self.params["classes"].tolist()

    @property
    def available(self):
        return self.bundle is not None

    def predict_features(self, features):
        """StandardScaler + logistic regression, using stable NumPy inference."""
        if not self.available:
            return None
        p = self.params
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != len(p["mean"]) or not np.isfinite(x).all():
            raise ValueError("分类特征维度不匹配或包含无效值")
        logits = ((x - p["mean"]) / np.where(p["scale"] == 0, 1, p["scale"])) @ p["coef"].T + p["intercept"]
        if logits.shape[1] == 1:
            positive = 1.0 / (1.0 + np.exp(-np.clip(logits[0, 0], -700, 700)))
            probabilities = np.array([1 - positive, positive])
        elif p["ovr"]:
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits[0], -700, 700)))
            probabilities /= probabilities.sum()
        else:
            probabilities = np.exp(logits[0] - logits[0].max())
            probabilities /= probabilities.sum()
        probs = {str(label): float(value) for label, value in zip(p["classes"], probabilities)}
        return {"label": str(p["classes"][probabilities.argmax()]),
                "confidence": float(probabilities.max()), "probs": probs}

    def predict(self, det, image, image_features=None):
        """det: 已加载的 WinCLIPPatch；image: 路径或PIL。
        返回 {"label":str, "confidence":float, "probs":{label:prob}}"""
        if not self.available:
            return None
        if image_features is not None:
            return self.predict_features(image_features)
        import torch
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        t = det.preprocess(image).unsqueeze(0).to(det.device)
        with torch.no_grad():
            f = det.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        feat = f[0].cpu().numpy().reshape(1, -1)
        return self.predict_features(feat)


if __name__ == "__main__":
    det = None
    print("分类器封装自检：模块可独立加载，predict 需传入 WinCLIPPatch 实例")
