# -*- coding: utf-8 -*-
"""
WinCLIP patch 级异常检测器（对齐先导实验，AUROC=0.9868）
========================================================
- 记忆库：正常样本的 patch 级特征（ViT-B-16 196 个 patch token）
- 异常分数：1 - patch 与记忆库最大余弦相似度，整图分数取 patch 最大值
- 热力图：14×14 patch 级异常分数上采样到原图
"""
import os
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import open_clip

ROOT = r"D:\programme file\ResearchArticle"
_WEIGHTS = os.path.join(ROOT, "weights", "CLIP-ViT-B-16-open_clip_model.safetensors")
N_PATCH = 14  # 224/16


class WinCLIPPatch:
    def __init__(self, model_name="ViT-B-16", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weight = _WEIGHTS if os.path.exists(_WEIGHTS) else None
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=weight)
        self.model = self.model.to(self.device).eval()
        self.bank = None          # [M, D]
        self.bank_info = None
        self._patch_cache = {}

        # hook 抓取 patch token（去掉 cls）
        self._patch_feats = {}
        handle = self.model.visual.transformer.resblocks[-1].register_forward_hook(
            self._hook)

    def _hook(self, module, input, output):
        self._patch_feats["value"] = output[:, 1:, :]

    # ---------- 文本零样本分类 ----------
    # 金属缺陷类型提示词（英文，CLIP训练语料以英文为主效果更好）
    DEFECT_PROMPTS = {
        "正常": "a photo of a normal undamaged metal part",
        "划痕": "a photo of a scratched metal surface with linear marks",
        "弯曲": "a photo of a bent deformed metal part",
        "色差": "a photo of a discolored stained metal surface",
        "裂纹": "a photo of a cracked fractured metal surface",
        "凸起": "a photo of a raised bump on metal surface",
        "凹陷": "a photo of a dent depression on metal surface",
    }

    @torch.no_grad()
    def classify_defect(self, image):
        """零样本缺陷类型分类：图像嵌入与各文本提示嵌入算余弦相似度。
        返回 {类型: 概率} 的字典，按概率降序。"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        img_feat = self.model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        text_tokens = open_clip.tokenize(list(self.DEFECT_PROMPTS.values())).to(self.device)
        txt_feats = self.model.encode_text(text_tokens)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

        sims = (img_feat @ txt_feats.T)[0].cpu().numpy()  # [7]
        # softmax转概率
        import numpy as np
        exp = np.exp(sims * 100)  # 温度系数放大差异
        probs = exp / exp.sum()
        result = dict(zip(self.DEFECT_PROMPTS.keys(), probs.tolist()))
        return dict(sorted(result.items(), key=lambda x: -x[1]))

    @torch.no_grad()
    def _get_patch_feats(self, img):
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        _ = self.model.encode_image(img_tensor)
        feats = self._patch_feats["value"][0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()               # [196, D]

    # ---------- 建库（带磁盘缓存） ----------
    def build_bank(self, normal_dir, max_samples=80, cache_path=None):
        """构建 patch 记忆库；cache_path 存在时直接加载，避免重复建库"""
        if cache_path and os.path.exists(cache_path):
            self.bank = np.load(cache_path)
            self.bank_info = {"n_images": max_samples,
                              "n_feats": self.bank.shape[0], "cached": True}
            return self.bank_info
        files = sorted(f for f in os.listdir(normal_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))[:max_samples]
        all_feats = []
        for f in files:
            img = Image.open(os.path.join(normal_dir, f)).convert("RGB")
            all_feats.append(self._get_patch_feats(img))
        self.bank = np.concatenate(all_feats, axis=0)   # [N*196, D]
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, self.bank)
        self.bank_info = {"n_images": len(files), "n_feats": self.bank.shape[0]}
        return self.bank_info

    # ---------- 检测 ----------
    @torch.no_grad()
    def detect(self, image):
        if self.bank is None:
            raise RuntimeError("请先 build_bank 构建记忆库")
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        feats = self._get_patch_feats(image)              # [196, D]
        sims = feats @ self.bank.T                        # [196, M]
        max_sims = sims.max(axis=1)                       # [196]
        anomaly = 1.0 - max_sims                          # [196]
        score = float(anomaly.max())                      # 整图分数 = patch 最大
        anomaly_map = anomaly.reshape(N_PATCH, N_PATCH)   # [14, 14]

        # 热力图叠加原图
        heat = self._overlay(image, anomaly_map)
        # 零样本缺陷类型分类
        try:
            cls = self.classify_defect(image)
        except Exception:
            cls = {}
        return {
            "score": score,
            "anomaly_map": anomaly_map,
            "heatmap": heat,
            "patch_max": float(max_sims.max()),
            "bank": self.bank_info,
            "classification": cls,
        }

    def _overlay(self, image, anomaly_map):
        """半透明红色热力图叠加在原图上，缺陷位置醒目"""
        canvas = image.convert("RGB").copy()
        W, H = canvas.size
        hm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        hm_img = Image.fromarray((hm * 255).astype(np.uint8)).resize(
            (W, H), Image.BILINEAR)
        heat_rgba = hm_img.convert("RGBA")
        r, g, b, a = heat_rgba.split()
        # 工业红热力图：红通道高，绿/蓝压低，alpha随热度
        r = r.point(lambda v: min(255, int(v * 2.2)))
        g = g.point(lambda v: int(v * 0.25))
        b = b.point(lambda v: int(v * 0.25))
        a = a.point(lambda v: int(v * 0.55))  # 透明度55%，更醒目
        overlay = Image.merge("RGBA", (r, g, b, a))
        return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


if __name__ == "__main__":
    det = WinCLIPPatch("ViT-B-16")
    print("建库:", det.build_bank(
        os.path.join(ROOT, "data", "mvtec", "metal_nut", "train", "good"), 80))
    import time
    for name in ["00_正常件.png", "01_弯曲缺陷_高置信.png",
                 "02_划痕缺陷_低置信.png", "03_色差缺陷.png"]:
        t0 = time.time()
        r = det.detect(os.path.join(ROOT, "data", "demo_samples", name))
        print(f"{name:<26} score={r['score']:.4f} ({time.time()-t0:.1f}s)")
