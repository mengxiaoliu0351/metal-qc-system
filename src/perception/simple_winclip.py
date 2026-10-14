# -*- coding: utf-8 -*-
"""
SimpleWinCLIP：真正的零样本异常检测器
=====================================
核心思想（对齐 WinCLIP 论文）：
  1. 用正常样本（train/good）切块编码，构建正常特征记忆库；
  2. 待检图切同样的块，逐块与正常库比对余弦相似度；
  3. 相似度低 = 异常；逐块异常分数绘制热力图；
  4. 全图异常分数 = 缺陷块的异常分数聚合（max / top-k mean）。

用法：
    detector = SimpleWinCLIP(model_name="ViT-B-16")
    detector.build_bank(normal_dir, n_images=20, grid=3)   # 建库
    result = detector.detect(image_path, grid=3)            # 检测
"""
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

# Windows torch OpenMP 冲突规避
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import open_clip

ROOT = r"D:\programme file\ResearchArticle"
_WEIGHTS = os.path.join(ROOT, "weights", "CLIP-ViT-B-16-open_clip_model.safetensors")


class SimpleWinCLIP:
    def __init__(self, model_name: str = "ViT-B-16", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weight = _WEIGHTS if os.path.exists(_WEIGHTS) else None
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=weight)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.bank = None          # [N, D] 正常特征库
        self.bank_info = None
        # 文本分支（WinCLIP 多模态）：正常 vs 缺陷文本模板
        self._text_feats = None

    # ---------- 建库 ----------
    @torch.no_grad()
    def build_bank(self, normal_dir, n_images=20, grid=3, seed=0):
        """normal_dir 下取前 n_images 张正常图，切 grid×grid 块入特征库"""
        files = sorted(f for f in os.listdir(normal_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))[:n_images]
        feats = []
        for f in files:
            img = Image.open(os.path.join(normal_dir, f)).convert("RGB")
            patches = self._crop_grid(img, grid)
            feats.append(self._encode(patches))          # [grid², D]
        self.bank = torch.cat(feats, dim=0)               # [N*grid², D]
        self.bank = torch.nn.functional.normalize(self.bank, dim=-1)
        self.bank_info = {"n_images": len(files), "grid": grid,
                          "dir": normal_dir, "n_feats": self.bank.shape[0]}
        return self.bank_info

    def _crop_grid(self, img, grid, overlap=0.1):
        W, H = img.size
        step_x, step_y = W / grid, H / grid
        ov_x, ov_y = step_x * overlap, step_y * overlap
        patches = []
        for iy in range(grid):
            for ix in range(grid):
                x0 = max(0, int(ix * step_x - ov_x))
                y0 = max(0, int(iy * step_y - ov_y))
                x1 = min(W, int((ix + 1) * step_x + ov_x))
                y1 = min(H, int((iy + 1) * step_y + ov_y))
                patches.append(img.crop((x0, y0, x1, y1)).convert("RGB"))
        return patches

    @torch.no_grad()
    def _encode(self, patches):
        batch = torch.stack([self.preprocess(p) for p in patches]).to(self.device)
        feats = self.model.encode_image(batch)
        return torch.nn.functional.normalize(feats, dim=-1).cpu()

    # ---------- 文本分支 ----------
    @torch.no_grad()
    def _build_text_feats(self, cls_name="metal nut"):
        """WinCLIP 文本分支：good vs damaged 模板对"""
        templates = [
            "a photo of a good {}.",
            "a photo of a perfect {}.",
            "a clean {}.",
            "a normal {}.",
            "a photo of a damaged {}.",
            "a photo of a broken {}.",
            "a defective {}.",
            "a scratched {}.",
        ]
        texts = [t.format(cls_name) for t in templates]
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = torch.nn.functional.normalize(feats, dim=-1).cpu()
        # good 类 = 前4个模板均值；damaged 类 = 后4个模板均值
        good = feats[:4].mean(0)
        damaged = feats[4:].mean(0)
        self._text_feats = torch.stack([good, damaged])   # [2, D]
        return self._text_feats

    # ---------- 检测 ----------
    @torch.no_grad()
    def detect(self, image, grid=3, alpha=0.5, k=3):
        """多模态异常检测：视觉库分支 + 文本分支融合
        score = alpha * (1 - 视觉kNN相似度) + (1-alpha) * (1 - good文本相似度)
        """
        if self.bank is None:
            raise RuntimeError("请先调用 build_bank 构建正常特征库")
        if self._text_feats is None:
            self._build_text_feats()
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        patches = self._crop_grid(image, grid)
        feats = self._encode(patches)                      # [grid², D]

        # 视觉分支：与正常库 top-k 近邻均值相似度
        sims = feats @ self.bank.t()                       # [grid², N]
        topk = torch.topk(sims, k=min(k, sims.shape[1]), dim=1).values
        vis_sim = topk.mean(dim=1)                         # [grid²]
        vis_score = 1.0 - vis_sim

        # 文本分支：good 文本相似度（低 = 异常）
        text_sims = feats @ self._text_feats.t()           # [grid², 2]
        txt_score = 1.0 - text_sims[:, 0]                  # good 文本相似度低→异常

        # 融合
        cell_scores = alpha * vis_score + (1 - alpha) * txt_score

        # 聚合：全图分数 = 最异常块的分数（max 对局部缺陷敏感）
        global_score = float(cell_scores.max())

        # 热力图：块级异常分数画网格
        heat = self._draw_heatmap(image, cell_scores, grid)
        return {
            "score": float(global_score),
            "cell_scores": cell_scores.numpy(),
            "vis_score": vis_score.numpy(),
            "txt_score": txt_score.numpy(),
            "heatmap": heat,
            "grid": grid,
            "bank": self.bank_info,
        }

    def _draw_heatmap(self, image, cell_scores, grid):
        canvas = image.convert("RGB").copy()
        W, H = canvas.size
        step_x, step_y = W / grid, H / grid
        cmap = self._make_cmap()
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for iy in range(grid):
            for ix in range(grid):
                s = float(cell_scores[iy * grid + ix])
                color = cmap[min(255, int(s * 255))]
                alpha = int(60 + 150 * min(1.0, s))
                x0, y0 = int(ix * step_x), int(iy * step_y)
                od.rectangle([x0, y0, x0 + int(step_x), y0 + int(step_y)],
                             fill=(color[0], color[1], color[2], alpha))
        return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    @staticmethod
    def _make_cmap():
        """蓝->黄->红 热力色表"""
        cmap = []
        for i in range(256):
            t = i / 255
            if t < 0.5:
                r, g, b = int(255 * t * 2), int(255 * t * 2), 255
            else:
                r, g, b = 255, int(255 * (1 - (t - 0.5) * 2)), int(255 * (1 - (t - 0.5) * 2))
            cmap.append((r, g, b))
        return cmap


if __name__ == "__main__":
    det = SimpleWinCLIP("ViT-B-16")
    info = det.build_bank(r"D:\programme file\ResearchArticle\data\mvtec\metal_nut\train\good",
                          n_images=20, grid=3)
    print("正常特征库:", info)
    import time
    for name in ["00_正常件.png", "01_弯曲缺陷_高置信.png",
                 "02_划痕缺陷_低置信.png", "03_色差缺陷.png"]:
        t0 = time.time()
        r = det.detect(os.path.join(ROOT, "data", "demo_samples", name), grid=3)
        print(f"{name:<24} score={r['score']:.4f} ({time.time()-t0:.1f}s)")
