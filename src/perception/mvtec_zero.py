# -*- coding: utf-8 -*-
"""
MVTec metal_nut 零样本泛化测试（知识梯度"正向维度"的裁决实验）
在金属螺母 4 类缺陷（bent/color/flip/scratch）上，检验紧固件域知识提示 L1/L2/L3
是否比现象级提示更有效——若有效，则论文故事为"知识在无先验的纹理缺陷上有害、
在有视觉先验的物体缺陷上有效"（边界双面刻画）。

用法：python mvtec_zero.py [--model ViT-B-16|ViT-L-14]
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import glob

import numpy as np
import open_clip
import torch
from PIL import Image
from torchvision import transforms

from prompts import L0a, L0b
from zeroshot_gate import DEFAULT_WEIGHTS

MV_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mvtec", "metal_nut")
MV_CLASSES = ["bent", "color", "flip", "scratch"]

# 紧固件域同构提示（现象 -> 成因 -> 机制风险；增量词为机制/工艺词）
MV_L1 = {
    "bent":   "a photo of a metal nut that is bent",
    "color":  "a photo of a metal nut with abnormal color",
    "flip":   "a photo of a metal nut assembled upside down",
    "scratch": "a photo of a metal nut with a surface scratch",
}
MV_L2 = {
    "bent":   "a photo of a metal nut that is bent, which is caused by over-torque or cross-threading during assembly",
    "color":  "a photo of a metal nut with abnormal color, which is caused by oxidation or heat exposure during processing",
    "flip":   "a photo of a metal nut assembled upside down, which is caused by an assembly error or misplacement",
    "scratch": "a photo of a metal nut with a surface scratch, which is caused by handling friction or tool contact",
}
MV_L3 = {
    "bent":   "a photo of a metal nut bent by over-torque, which induces stress concentration and reduces the clamping force",
    "color":  "a photo of a metal nut with color change caused by oxidation, which indicates surface degradation and corrosion risk",
    "flip":   "a photo of a metal nut flipped by assembly error, which leads to functional failure of the fastening pair",
    "scratch": "a photo of a metal nut scratched by tool contact, which induces stress concentration and fatigue failure risk",
}


def mask_to_bbox(mask_path):
    """掩码 -> 缺陷 bbox（非零像素的包围盒），返回 (x1,y1,x2,y2)。"""
    m = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.nonzero(m > 0)
    if len(xs) == 0:
        return None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = 8
    return (max(0, x1 - pad), max(0, y1 - pad),
            min(m.shape[1], x2 + pad), min(m.shape[0], y2 + pad))


def load_mvtec():
    """返回 {cls: [(img_path, bbox), ...]}，按掩码裁剪。"""
    data = {c: [] for c in MV_CLASSES}
    for cls in MV_CLASSES:
        imgs = sorted(glob.glob(os.path.join(MV_ROOT, "test", cls, "*.png")))
        for img_path in imgs:
            base = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(MV_ROOT, "ground_truth", cls, f"{base}_mask.png")
            if not os.path.exists(mask_path):
                continue
            bbox = mask_to_bbox(mask_path)
            if bbox is None:
                continue
            data[cls].append((img_path, bbox))
        print(f"  {cls:<10} {len(data[cls])} 个缺陷裁剪")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-16")
    args = ap.parse_args()

    _w = DEFAULT_WEIGHTS.get(args.model)
    pt = _w if _w and os.path.exists(_w) else None
    print(f"模型: {args.model} | 权重: {pt or '(在线)'}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=pt)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.eval()

    data = load_mvtec()
    feats, labels = [], []
    for cls in MV_CLASSES:
        for img_path, bbox in data[cls]:
            img = Image.open(img_path).convert("RGB").crop(bbox)
            feats.append(preprocess(img))
            labels.append(MV_CLASSES.index(cls))
    feats = torch.stack(feats)
    with torch.no_grad():
        feats = model.encode_image(feats)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    labels = torch.tensor(labels)

    groups = {"L1": MV_L1, "L2": MV_L2, "L3": MV_L3,
              "L0a": {c: [L0a] for c in MV_CLASSES},
              "L0b": {c: [L0b] for c in MV_CLASSES}}
    print("\n=== MVTec metal_nut 零样本（top-1%）===")
    print(f"{'组':<6}{'整体':>8}  " + "  ".join(f"{c:>8}" for c in MV_CLASSES))
    for name, prompts in groups.items():
        text_feats = []
        for cls in MV_CLASSES:
            with torch.no_grad():
                t = tokenizer(prompts[cls]).to(feats.device)
                f = model.encode_text(t)
            f = f / f.norm(dim=-1, keepdim=True)
            text_feats.append(f.mean(0))
        text_feats = torch.stack(text_feats)
        logits = feats @ text_feats.t()
        preds = logits.argmax(1)
        overall = (preds == labels).float().mean().item() * 100
        row = f"{name:<6}{overall:>8.1f}  "
        for i, cls in enumerate(MV_CLASSES):
            m = labels == i
            ca = (preds[m] == i).float().mean().item() * 100 if m.sum() else float("nan")
            row += f"{ca:>8.1f}"
        print(row)

    print("\n判读：若 L3 > L2 > L1 成立 -> 知识在 MVTec 上有效，论文可做'边界双面'故事；")
    print("      若仍 L1 > L3 -> 知识在所有被测场景均无益，走'知识有害+设计准则'叙事。")


if __name__ == "__main__":
    main()
