# -*- coding: utf-8 -*-
"""
few-shot 知识梯度主实验（TIP-Adapter 式训练免融合）
在 oracle 裁剪上，对每个提示组（L1/L2/L3/L0a/L0b/基线）和每个 K（0/1/2/4/8/full）做少样本分类。
机制：logits = alpha * cache_logits + text_logits
  - text_logits：测试图与文本提示的余弦（零样本部分，携带提示内容）
  - cache_logits：测试图与 K 个示例图的相似度按标签加权（少样本部分，提供锚点）
协议：固定测试集（每类固定 60 张）；训练子集嵌套采样；多 seeds 报 mean±std。
优化：每 seed 只编码一次（测试集 + 完整训练池），各 K 用切片，避免重复编码。

用法：
    python few_shot.py --data <NEU-DET> --model ViT-B-16 --max-per-class 120 --seeds 10
    # 快速冒烟：加 --quick（只用 K=0,2,8 和 1 个 seed）
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import random

import numpy as np
import open_clip
import torch
from PIL import Image
from torchvision import transforms

from prompts import build_prompt_sets, CLASSES
from zeroshot_gate import load_neu_det, DEFAULT_WEIGHTS

GROUPS = ["L1", "L2", "L3", "L0a", "L0b", "Baseline-A", "Baseline-B"]
TEST_PER_CLASS = 60  # 固定测试集（NEU-DET 6:2:2 中的 2）


def build_crop_samples(data):
    """把 load_neu_det 的结果转成固定测试集 + 训练池（按类）。"""
    rng = random.Random(2026)  # 固定种子：测试集划分在 seed 之间不变
    train_pool, test_set = {}, {}
    for c in CLASSES:
        items = list(data[c])
        rng.shuffle(items)
        n_test = min(TEST_PER_CLASS, len(items))
        test_set[c] = items[:n_test]
        train_pool[c] = items[n_test:]
    return train_pool, test_set


def encode_images(samples, model, preprocess, device):
    if not samples:
        return None, None
    feats, labels = [], []
    for img_path, bbox, cls in samples:
        img = Image.open(img_path).convert("RGB").crop(bbox)
        feats.append(preprocess(img))
        labels.append(cls)
    feats = torch.stack(feats).to(device)
    with torch.no_grad():
        feats = model.encode_image(feats)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats, torch.tensor(labels)


def encode_texts(model, tokenizer, texts, device):
    with torch.no_grad():
        t = tokenizer(texts).to(device)
        feats = model.encode_text(t)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def tip_adapter(test_feats, test_labels, train_feats, train_labels, text_feats,
                alpha=1.0, beta=5.5):
    """TIP-Adapter：alpha*cache_logits + text_logits，返回 top-1 准确率。"""
    C = text_feats.shape[0]
    text_logits = 100.0 * (test_feats @ text_feats.T)  # 线性放缩，避免指数淹没
    if train_feats is None:  # K=0：纯零样本
        preds = text_logits.argmax(1)
        return (preds == test_labels).float().mean().item()
    S = test_feats @ train_feats.T  # [Nt, N]
    cache_logits = torch.exp(((S - 1.0) * beta).clamp(max=100))
    onehot = torch.zeros(train_labels.shape[0], C).to(train_feats.device)
    onehot.scatter_(1, train_labels.view(-1, 1), 1.0)
    cache_logits = cache_logits @ onehot
    logits = alpha * cache_logits + text_logits
    preds = logits.argmax(1)
    return (preds == test_labels).float().mean().item()


def prepare_features(train_pool, test_set, model, preprocess, device, seed):
    """每 seed 只编码一次：完整训练池 + 固定测试集。"""
    rng = random.Random(seed)
    pool_samples, test_samples = [], []
    for c in CLASSES:
        p = list(train_pool[c])
        rng.shuffle(p)
        pool_samples += [(img, bbox, CLASSES.index(c)) for img, bbox in p]
        test_samples += [(img, bbox, CLASSES.index(c)) for img, bbox in test_set[c]]
    pool_feats, pool_labels = encode_images(pool_samples, model, preprocess, device)
    test_feats, test_labels = encode_images(test_samples, model, preprocess, device)
    counts = {c: len(train_pool[c]) for c in CLASSES}
    return pool_feats, pool_labels, test_feats, test_labels, counts


def eval_k(shots, pool_feats, pool_labels, test_feats, test_labels, counts,
           model, tokenizer, device, alpha):
    """对给定 K 评估所有提示组。"""
    k = None if shots == "full" else min(shots, min(counts.values()))
    sel = []
    start = 0
    for c in CLASSES:
        n = counts[c]
        take = n if k is None else k
        sel += list(range(start, start + take))
        start += n
    if not sel:
        train_feats = train_labels = None
    else:
        idx = torch.tensor(sel)
        train_feats = pool_feats[idx]
        train_labels = pool_labels[idx]

    results = {}
    for group in GROUPS:
        class_text = []
        for c in CLASSES:
            texts = build_prompt_sets()[group][c]
            feats = encode_texts(model, tokenizer, texts, device).mean(0)
            feats = feats / feats.norm()
            class_text.append(feats)
        text_feats = torch.stack(class_text)
        results[group] = tip_adapter(test_feats, test_labels, train_feats,
                                     train_labels, text_feats, alpha=alpha)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="ViT-B-16")
    ap.add_argument("--max-per-class", type=int, default=120,
                    help="每类最多用多少张（含训练池+测试集），须 >60")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="快速冒烟：K=0,2,8 + 1 seed")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--save", default=None, help="把结果存成 CSV（供 plots.py 用）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    K_LIST = [0, 1, 2, 4, 8, "full"] if not args.quick else [0, 2, 8]
    seeds = [0] if args.quick else range(args.seeds)

    _w = DEFAULT_WEIGHTS.get(args.model)
    pt = _w if _w and os.path.exists(_w) else None
    print(f"设备: {args.device} | 模型: {args.model} | 权重: {pt or '(在线)'}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=pt)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(args.device).eval()

    data = load_neu_det(args.data, max_images_per_class=args.max_per_class)
    train_pool, test_set = build_crop_samples(data)
    npool = len(train_pool["scratches"])
    ntest = len(test_set["scratches"])
    print(f"训练池: 每类 {npool} 张 | 固定测试: 每类 {ntest} 张 | seeds: {len(list(seeds))}")

    print("\n=== few-shot 知识梯度（top-1%，mean±std）===")
    header = "K".ljust(6) + "".join(g.ljust(12) for g in GROUPS)
    print(header)
    print("-" * len(header))
    csv_rows = []  # 每行 (K, group, mean, std)
    for shots in K_LIST:
        accs = {g: [] for g in GROUPS}
        for seed in seeds:
            p = prepare_features(train_pool, test_set, model, preprocess, args.device, seed)
            pool_feats, pool_labels, test_feats, test_labels, counts = p
            r = eval_k(shots, pool_feats, pool_labels, test_feats, test_labels,
                       counts, model, tokenizer, args.device, args.alpha)
            for g in GROUPS:
                accs[g].append(r[g] * 100)
        row = str(shots).ljust(6)
        for g in GROUPS:
            arr = np.array(accs[g])
            mean, std = arr.mean(), arr.std()
            row += (f"{mean:.1f}±{std:.1f}" if len(arr) > 1 else f"{mean:.1f}").ljust(12)
            csv_rows.append((str(shots), g, f"{mean:.1f}", f"{std:.1f}"))
        print(row)
    if args.save:
        import csv
        os.makedirs(os.path.dirname(args.save), exist_ok=True) if os.path.dirname(args.save) else None
        with open(args.save, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["K", "group", "mean", "std"])
            w.writerows(csv_rows)
        print(f"结果已存: {args.save}")


if __name__ == "__main__":
    main()
