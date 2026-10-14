# -*- coding: utf-8 -*-
"""
0-shot gate（R1 生死线，第一优先级）
在 NEU-DET 真值框裁剪（oracle-crop）上跑 CLIP 零样本分类，判定任务可行性。

安装依赖（Windows，命令行）：
    pip install torch torchvision open_clip_torch pillow numpy

运行：
    python zeroshot_gate.py --data <NEU-DET 根目录> --model ViT-B-16

NEU-DET 目录结构（官方 GitHub 下载）：
    <root>/IMAGES/        *.jpg ，共 1800 张，类名前缀命名（如 Scratches_1.jpg）
    <root>/ANNOTATIONS/   *.txt ，每张图一个，行格式见下方 parse 函数说明

判定标准（来自方案 R1）：
    top1 >= 60%  -> [绿色] 任务可行，直接进 few-shot 知识梯度实验
    30% <= top1 < 60% -> [黄色] 先调 prompt / 换 backbone（ViT-B-16 -> ViT-L-14）
    top1 < 30%   -> [红色] 停手，重新评估任务/换数据集，不硬推
"""
import os

# Windows 下 torch 与其他库的 OpenMP 冲突规避，必须在导入 torch 之前设置
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 国内访问 HuggingFace 权重走镜像（首次下载 CLIP 权重用；已缓存后不影响）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import glob
import re
import xml.etree.ElementTree as ET

import numpy as np
import open_clip
import torch
from PIL import Image

from prompts import build_prompt_sets, CLASSES

# 本地权重（镜像下载，避免在线拉取失败/404）
_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
DEFAULT_WEIGHTS = {
    "ViT-B-16": os.path.join(_WEIGHTS_DIR, "CLIP-ViT-B-16-open_clip_model.safetensors"),
    "ViT-L-14": os.path.join(_WEIGHTS_DIR, "CLIP-ViT-L-14-open_clip_model.safetensors"),
}

# NEU-DET 类名 -> 我们的类键（移除 rolled-in scale）
_NEU_NAME_TO_CLASS = {
    "crazing": "crazing", "inclusion": "inclusion", "patches": "patches",
    "pitted_surface": "pitted_surface", "pitted": "pitted_surface",
    "rolled-in_scale": "__skip__", "rolled-in": "__skip__", "rolled in": "__skip__",
    "scratches": "scratches", "scratch": "scratches",
}
# 若标注用数字类别 id（常见顺序：1=crazing ... 6=scratches）
_ID_TO_CLASS = {1: "crazing", 2: "inclusion", 3: "patches", 4: "pitted_surface",
                5: "__skip__", 6: "scratches"}


def parse_bbox(line: str, image_w: int = 200, image_h: int = 200):
    """解析 NEU-DET 标注行，返回 (class_key, x1, y1, x2, y2)。
    支持的两种常见格式：
      格式A：class_name x1 y1 x2 y2 x3 y3 x4 y4 （4 角点，旋转框）
      格式B：class_id  x1 y1 w h
    坐标若在 [0,1] 内按比例归一化处理。返回的 x1..y2 为像素坐标。
    """
    parts = line.split()
    if len(parts) < 5:
        return None

    first = parts[0]
    # 判断第一个 token 是类名还是数字 id
    if first.lower() in _NEU_NAME_TO_CLASS:
        cls_key = _NEU_NAME_TO_CLASS[first.lower()]
        coords = [float(p) for p in parts[1:]]
    elif re.fullmatch(r"\d+", first):
        cls_key = _ID_TO_CLASS.get(int(first), "__skip__")
        coords = [float(p) for p in parts[1:]]
    else:
        return None

    if cls_key == "__skip__":
        return ("__skip__", 0, 0, 0, 0)

    if len(coords) == 4:  # 格式B: x y w h
        x, y, w, h = coords
        if max(coords) <= 1.0:  # 归一化坐标
            x, y, w, h = x * image_w, y * image_h, w * image_w, h * image_h
        return (cls_key, int(x), int(y), int(x + w), int(y + h))

    if len(coords) == 8:  # 格式A: 4 角点
        xs = coords[0::2]
        ys = coords[1::2]
        if max(coords) <= 1.0:
            xs = [v * image_w for v in xs]
            ys = [v * image_h for v in ys]
        return (cls_key, int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    return None


def parse_voc_xml(ann_path: str):
    """解析 Pascal VOC 格式的 .xml 标注，返回 [(class_key, x1, y1, x2, y2), ...]。
    用于兼容 Kaggle/部分镜像的 .xml 标注版本。
    """
    results = []
    try:
        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name not in _NEU_NAME_TO_CLASS:
                continue
            cls_key = _NEU_NAME_TO_CLASS[name]
            if cls_key == "__skip__":
                continue
            bb = obj.find("bndbox")
            x1 = int(float(bb.findtext("xmin")))
            y1 = int(float(bb.findtext("ymin")))
            x2 = int(float(bb.findtext("xmax")))
            y2 = int(float(bb.findtext("ymax")))
            if x2 > x1 and y2 > y1:
                results.append((cls_key, x1, y1, x2, y2))
    except Exception as e:
        print(f"  [!] 解析 {ann_path} 失败: {e}")
    return results


def load_neu_det(root: str, max_images_per_class: int = 50):
    """加载 NEU-DET 并返回 {class_key: [(img_path, bbox_xyxy), ...]}。
    max_images_per_class 限制每类裁剪数，跑 gate 用子集即可（更快）。
    """
    img_dir = os.path.join(root, "IMAGES")
    ann_dir = os.path.join(root, "ANNOTATIONS")
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not images:
        images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not images:
        raise FileNotFoundError(f"未在 {img_dir} 找到图片，请检查 --data 路径")

    data = {c: [] for c in CLASSES}
    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        # 优先 .txt（原始格式），回退 .xml（Kaggle/部分镜像的 Pascal VOC 格式）
        ann_txt = os.path.join(ann_dir, base + ".txt")
        ann_xml = os.path.join(ann_dir, base + ".xml")
        if os.path.exists(ann_txt):
            with open(ann_txt, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parsed = parse_bbox(line)
                    if parsed is None:
                        continue
                    cls_key, x1, y1, x2, y2 = parsed
                    if cls_key == "__skip__":
                        continue
                    if x2 <= x1 or y2 <= y1:
                        continue
                    data[cls_key].append((img_path, (x1, y1, x2, y2)))
                    break  # 一张图取第一个缺陷框即可（gate 阶段）
        elif os.path.exists(ann_xml):
            for cls_key, x1, y1, x2, y2 in parse_voc_xml(ann_xml):
                data[cls_key].append((img_path, (x1, y1, x2, y2)))
                break

    print("加载统计（前几个类别）：")
    for c in CLASSES:
        print(f"  {c:<16} {len(data[c])} 个裁剪")
        if len(data[c]) == 0:
            print(f"    [!] 类别 {c} 没有裁剪！请核对 ANNOTATIONS 格式（运行 parse_bbox 的调试）")

    # 按类截断，保证各类样本均衡
    for c in CLASSES:
        data[c] = data[c][:max_images_per_class]
    return data


def encode_text(model, tokenizer, texts, device):
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        return model.encode_text(tokens)


def run_zero_shot(data, model, preprocess, tokenizer, device):
    """对每个提示组计算 top-1 准确率（每类各取 max 提示）。"""
    prompt_sets = build_prompt_sets()
    groups = ["L1", "L2", "L3", "L0a", "L0b", "Baseline-A", "Baseline-B"]

    # 预计算所有裁剪的图像特征
    img_feats, img_labels = [], []
    for cls_key in CLASSES:
        for img_path, bbox in data[cls_key]:
            img = Image.open(img_path).convert("RGB").crop(bbox)
            img_feats.append(preprocess(img).unsqueeze(0))
            img_labels.append(CLASSES.index(cls_key))
    img_feats = torch.cat(img_feats).to(device)
    img_labels = torch.tensor(img_labels)
    with torch.no_grad():
        img_feats = model.encode_image(img_feats)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

    results = {}
    print("\n" + "=" * 70)
    print(f"0-shot gate 结果（{len(img_feats)} 个裁剪，ViT-B-16，固定测试集）")
    print("=" * 70)
    for group in groups:
        # 对每个类取该组所有提示的文本特征，逐类 softmax
        class_text_feats = []
        for cls_key in CLASSES:
            texts = prompt_sets[group][cls_key]
            feats = encode_text(model, tokenizer, texts, device)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            class_text_feats.append(feats.mean(dim=0))  # 类内提示平均
        class_text_feats = torch.stack(class_text_feats)  # [num_classes, D]
        logits = img_feats @ class_text_feats.t()  # [N, num_classes]
        preds = logits.argmax(dim=1)
        acc = (preds == img_labels).float().mean().item() * 100
        results[group] = acc
        marker = "L" if group in ("L3",) else " "
        print(f"  {group:<14} top-1 = {acc:5.1f}%")
    return results


def gate_verdict(results):
    l1, l2, l3 = results["L1"], results["L2"], results["L3"]
    gradient_ok = (l3 > l2 > l1) and (l3 - l1 >= 2.0)
    print("\n" + "=" * 70)
    print("判定（R1 三档）")
    print("=" * 70)
    if l3 >= 60 and gradient_ok:
        print(f"  [绿色] L3={l3:.1f}% >= 60% 且 L3>L2>L1：任务可行，知识梯度成立，直接进 few-shot 实验。")
    elif l3 >= 30:
        print(f"  [黄色] L3={l3:.1f}% 落在 30-60%：先调 prompt / 换 backbone（ViT-B-16 -> ViT-L-14）再进。")
    else:
        print(f"  [红色] L3={l3:.1f}% < 30%：CLIP 在 NEU-DET 上接近随机，停手重新评估任务/换数据集。")
    if not gradient_ok:
        print("  [!] 知识梯度方向未满足（L3>L2>L1），请检查提示文本与类别语义是否自洽。")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="NEU-DET 根目录（含 IMAGES/ 与 ANNOTATIONS/）")
    parser.add_argument("--model", default="ViT-B-16", help="open_clip 模型名")
    parser.add_argument("--max-per-class", type=int, default=50, help="每类最多用多少个裁剪")
    parser.add_argument("--pretrained", default=None,
                        help="本地权重路径；默认用 code/weights/ 下的镜像权重，找不到才在线拉")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _default = DEFAULT_WEIGHTS.get(args.model)
    pt = args.pretrained or (_default if _default and os.path.exists(_default) else None)
    if pt:
        print(f"使用本地权重: {pt}")
    print(f"设备: {args.device} | 模型: {args.model}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=pt)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(args.device).eval()

    data = load_neu_det(args.data, max_images_per_class=args.max_per_class)
    results = run_zero_shot(data, model, preprocess, tokenizer, args.device)
    gate_verdict(results)


if __name__ == "__main__":
    main()
