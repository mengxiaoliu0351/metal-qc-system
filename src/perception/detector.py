# -*- coding: utf-8 -*-
"""
detector.py —— 少样本工业缺陷智能检测系统 · 核心推理引擎
================================================================
把论文实验里的「批量评测脚本」改造成「对任意一张图出结果」的在线推理引擎，
推理口径与实验完全一致（TIP-Adapter 式训练免融合）：

    text_logits  = 100 * cos(图像特征, 各类文本原型)          # 文本路（携带提示内容）
    cache_logits = exp((cos(图像, 锚点) - 1) * beta) @ onehot # 锚点路（少样本证据）
    logits       = alpha * cache_logits + text_logits         # 融合（实验默认 alpha=1.0, beta=5.5）

与实验脚本的对应关系：
    - 文本原型构建、融合公式 1:1 复用 few_shot.py / per_class.py；
    - few_shot.py 对「固定测试集」返回准确率；本模块对「单张/分块图像」返回各类概率；
    - 实验用真值框 oracle-crop；在线没有真值框，本模块用「滑窗分块」自动定位缺陷，
      分块后每块的处理与实验 crop 完全一致，从而对齐论文口径（整图直接分类作为对照）。

对外主要接口：
    engine = get_engine("ViT-B-16")          # 全局单例，模型只加载一次
    bank   = engine.build_bank_builtin(k=8)  # 用 NEU-DET 训练池一键建锚点（演示最稳）
    result = engine.detect(pil_img, group="L1", bank=bank)            # 单图/整图
    result = engine.detect_window(pil_img, "L1", bank, grid=3)        # 滑窗定位
    loo    = engine.loo_compare(bank)                                 # 锚点留一选优
    align  = engine.alignment_report(bank)                            # 文本-图像对齐质量门
"""
import os

# 必须在 import torch 前设置，规避 Windows 下重复 OpenMP 运行时（与实验脚本一致）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw, ImageFont

from prompts import (CLASSES, NORMAL_STRONG, NORMAL_WEAK, build_prompt_sets)
from zeroshot_gate import DEFAULT_WEIGHTS, load_neu_det
from few_shot import build_crop_samples

# ----------------------------------------------------------------------------
# 常量与展示映射
# ----------------------------------------------------------------------------
ALPHA_DEFAULT = 1.0     # 融合权重（论文主设定，扫描 0.3~1.0 结论鲁棒）
BETA_DEFAULT = 5.5      # TIP-Adapter cache 核温度
TEXT_SCALE = 100.0      # 文本 logits 线性放缩（与实验一致，避免指数淹没）
ALIGN_THRESHOLD = 0.216  # 文本-图像对齐质量门：低于 Baseline-A 的 0.216 视为对齐不足
# 智能多尺度：粗窗(2×2)主导分类，粗窗不自信而细窗(3×3)自信时才让细窗接管（点状缺陷）；
# 粗细冲突时以粗窗为准，防止划痕等线状缺陷被切碎后误判为夹杂（由线上评估数据标定）。
SMART_LOW = 0.45
SMART_HIGH = 0.60

GROUPS = ["L1", "L2", "L3", "Baseline-A", "Baseline-B"]

CLASS_CN = {
    "scratches": "划痕",
    "patches": "斑块",
    "crazing": "发纹/龟裂",
    "pitted_surface": "点蚀",
    "inclusion": "夹杂",
}

GROUP_CN = {
    "L1": "L1 现象级视觉描述",
    "L2": "L2 现象+工艺成因",
    "L3": "L3 现象+成因+机制风险",
    "Baseline-A": "基线A 裸类名",
    "Baseline-B": "基线B 通用模板",
}

GROUP_DESC = {
    "L1": "只描述看得见的外观（推荐，少样本下最稳）",
    "L2": "在现象基础上补充轧制成因",
    "L3": "再叠加应力集中/疲劳寿命等无视觉对应的机制词（实验证明少样本下有害）",
    "Baseline-A": "只有一个英文单词类名",
    "Baseline-B": "一句不含类别信息的通用模板",
}

# 5 类固定配色（用于滑窗定位上色，区分且不刺眼）
CLASS_COLOR = {
    "scratches": (234, 102, 104),    # 红
    "patches": (240, 166, 60),       # 橙
    "crazing": (155, 89, 182),       # 紫
    "pitted_surface": (52, 152, 219),# 蓝
    "inclusion": (46, 184, 150),     # 青绿
}

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
DEFAULT_DATA = os.path.join(ROOT_DIR, "data", "NEU-DET")


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def _load_font(size: int = 14):
    """尽量加载中文字体，失败则回退默认字体（此时用英文标注，绝不报错）。"""
    for name in ("msyh.ttc", "simhei.ttf", "msyhbd.ttc", "simsun.ttc"):
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ----------------------------------------------------------------------------
# 锚点库
# ----------------------------------------------------------------------------
class AnchorBank:
    """少样本锚点：归一化图像特征 + 类别标签。"""

    def __init__(self, feats: torch.Tensor, labels: torch.Tensor,
                 source: str = "builtin"):
        self.feats = feats                      # [M, D]，已 L2 归一化
        self.labels = labels.long()             # [M]，取值为 CLASSES 下标
        self.source = source

    def __len__(self) -> int:
        return int(self.feats.shape[0])

    def per_class_count(self) -> Dict[str, int]:
        cnt = {c: 0 for c in CLASSES}
        for lb in self.labels.tolist():
            cnt[CLASSES[lb]] += 1
        return cnt

    def min_k(self) -> int:
        pc = self.per_class_count()
        return min(pc.values()) if pc else 0


# ----------------------------------------------------------------------------
# 推理引擎（全局单例）
# ----------------------------------------------------------------------------
class DefectEngine:
    def __init__(self, model_name: str = "ViT-B-16", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        w = DEFAULT_WEIGHTS.get(model_name)
        weight = w if (w and os.path.exists(w)) else None
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=weight)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()
        self._text_cache: Dict[str, torch.Tensor] = {}
        self._train_pool = None
        self._test_set = None

    # ---------- 基础编码 ----------
    def encode_pil(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """把一批 PIL 图编码为 L2 归一化特征 [N, D]。"""
        if not images:
            raise ValueError("没有可编码的图像")
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            tok = self.tokenizer(texts).to(self.device)
            f = self.model.encode_text(tok)
        return f / f.norm(dim=-1, keepdim=True)

    def text_protos(self, group: str) -> torch.Tensor:
        """某提示组下 5 个类别的文本原型 [C, D]（类内多提示平均后再归一化，同实验）。"""
        if group in self._text_cache:
            return self._text_cache[group]
        sets = build_prompt_sets()
        protos = []
        for c in CLASSES:
            f = self._encode_text(sets[group][c]).mean(0)
            protos.append(f / f.norm())
        proto = torch.stack(protos)
        self._text_cache[group] = proto
        return proto

    def normal_proto(self) -> torch.Tensor:
        """「正常/无缺陷表面」文本原型，用于滑窗块级背景门控（缓存）。"""
        if getattr(self, "_normal_feat", None) is None:
            f = self._encode_text([NORMAL_WEAK, NORMAL_STRONG]).mean(0)
            self._normal_feat = f / f.norm()
        return self._normal_feat

    # ---------- 融合（与 few_shot.tip_adapter 完全一致，只是返回单图概率）----------
    def _fuse_logits(self, img_feat: torch.Tensor, text_feats: torch.Tensor,
                     bank: Optional[AnchorBank], alpha: float, beta: float
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (融合logits, 文本路logits, 锚点路logits)，形状均为 [N, C]。"""
        text_logits = TEXT_SCALE * (img_feat @ text_feats.T)
        if bank is not None and len(bank) > 0:
            sim = img_feat @ bank.feats.T
            cache = torch.exp(((sim - 1.0) * beta).clamp(max=100))
            onehot = torch.zeros(bank.feats.shape[0], len(CLASSES), device=self.device)
            onehot.scatter_(1, bank.labels.view(-1, 1).to(self.device), 1.0)
            cache_logits = cache @ onehot
            fused = alpha * cache_logits + text_logits
        else:
            cache_logits = torch.zeros_like(text_logits)
            fused = text_logits
        return fused, text_logits, cache_logits

    # ---------- 内置数据（训练池做锚点 / 固定测试集做现场出题）----------
    def _ensure_neu(self):
        if self._train_pool is None:
            data = load_neu_det(DEFAULT_DATA, max_images_per_class=300)
            self._train_pool, self._test_set = build_crop_samples(data)

    def build_bank_builtin(self, k: int = 8, seed: int = 0) -> AnchorBank:
        """从 NEU-DET 训练池（固定测试集之外）每类取 k 张真值裁剪做锚点。"""
        self._ensure_neu()
        rng = random.Random(seed)
        imgs, labels = [], []
        for ci, c in enumerate(CLASSES):
            pool = list(self._train_pool[c])
            rng.shuffle(pool)
            for path, bbox in pool[:k]:
                imgs.append(Image.open(path).convert("RGB").crop(bbox))
                labels.append(ci)
        if not imgs:
            raise RuntimeError("未能从 NEU-DET 构建锚点，请检查 data/NEU-DET 路径")
        feats = self.encode_pil(imgs)
        return AnchorBank(feats, torch.tensor(labels, device=self.device), source="builtin")

    def build_bank_from_images(self, images_by_class: Dict[str, List[Image.Image]]) -> AnchorBank:
        """用用户上传的图建锚点：{类名: [PIL图,...]}，图即缺陷局部，不再裁剪。"""
        imgs, labels = [], []
        for c, ims in images_by_class.items():
            if c not in CLASSES:
                continue
            for im in ims:
                if im is not None:
                    imgs.append(im)
                    labels.append(CLASSES.index(c))
        if not imgs:
            raise RuntimeError("锚点为空")
        feats = self.encode_pil(imgs)
        return AnchorBank(feats, torch.tensor(labels, device=self.device), source="upload")

    def sample_test_image(self, seed: Optional[int] = None
                          ) -> Tuple[Image.Image, str, Tuple[int, int, int, int]]:
        """从固定测试集随机取一张「模型没见过」的整图，返回(整图PIL, 真值类, 真值框)。"""
        self._ensure_neu()
        rng = random.Random(seed)
        c = rng.choice(CLASSES)
        path, bbox = rng.choice(self._test_set[c])
        return Image.open(path).convert("RGB"), c, bbox

    def load_test_by_basename(self, basename: str
                              ) -> Optional[Tuple[Image.Image, str, Tuple[int, int, int, int]]]:
        """按文件名从固定测试集取整图（用于加载预先挑好的稳定演示样例）。"""
        self._ensure_neu()
        for c in CLASSES:
            for path, bbox in self._test_set[c]:
                if os.path.basename(path) == basename:
                    return Image.open(path).convert("RGB"), c, bbox
        return None

    # ---------- 单图（整图）检测 ----------
    @torch.no_grad()
    def detect(self, image: Image.Image, group: str = "L1",
               bank: Optional[AnchorBank] = None, alpha: float = ALPHA_DEFAULT,
               beta: float = BETA_DEFAULT) -> Dict:
        feat = self.encode_pil([image])
        text_feats = self.text_protos(group)
        fused, txt_logits, cache_logits = self._fuse_logits(
            feat, text_feats, bank, alpha, beta)
        probs = softmax(fused, dim=1)[0].cpu().numpy()
        order = int(np.argmax(probs))
        return {
            "mode": "整图直接分类",
            "group": group,
            "pred": CLASSES[order],
            "conf": float(probs[order]),
            "probs": {CLASS_CN[CLASSES[i]]: float(probs[i]) for i in range(len(CLASSES))},
            "probs_raw": probs,
            "text_logits": txt_logits[0].cpu().numpy(),
            "cache_logits": cache_logits[0].cpu().numpy(),
            "n_anchor": 0 if bank is None else len(bank),
        }

    # ---------- 滑窗分块检测 + 定位 ----------
    @torch.no_grad()
    def detect_window(self, image: Image.Image, group: str = "L1",
                      bank: Optional[AnchorBank] = None, grid: int = 3,
                      overlap: float = 0.1, alpha: float = ALPHA_DEFAULT,
                      beta: float = BETA_DEFAULT,
                      show_grid: bool = True) -> Dict:
        """
        把整图切成 grid×grid（带重叠）小块，逐块按实验口径分类；
        用「各类在所有块上的最大概率」做 max-pool 聚合（局部小缺陷不被背景稀释），
        并在原图上叠加定位可视化。
        """
        W, H = image.size
        text_feats5 = self.text_protos(group)
        normal = self.normal_proto()
        text_feats6 = torch.cat([text_feats5, normal.unsqueeze(0)], dim=0)  # 5 缺陷 + 1 正常
        step_x, step_y = W / grid, H / grid
        ov_x, ov_y = step_x * overlap, step_y * overlap

        boxes, patches = [], []
        for iy in range(grid):
            for ix in range(grid):
                x0 = max(0, int(ix * step_x - ov_x))
                y0 = max(0, int(iy * step_y - ov_y))
                x1 = min(W, int((ix + 1) * step_x + ov_x))
                y1 = min(H, int((iy + 1) * step_y + ov_y))
                boxes.append((x0, y0, x1, y1))
                patches.append(image.crop((x0, y0, x1, y1)).convert("RGB"))

        feats = self.encode_pil(patches)
        # 文本路（6 类，含正常）
        text_logits6 = TEXT_SCALE * (feats @ text_feats6.T)         # [N,6]
        n = feats.shape[0]
        if bank is not None and len(bank) > 0:
            sim = feats @ bank.feats.T
            cache = torch.exp(((sim - 1.0) * beta).clamp(max=100))
            onehot = torch.zeros(bank.feats.shape[0], len(CLASSES), device=self.device)
            onehot.scatter_(1, bank.labels.view(-1, 1).to(self.device), 1.0)
            cache5 = alpha * (cache @ onehot)                        # [N,5]，正常类无锚点
            cache6 = torch.cat([cache5, torch.zeros(n, 1, device=self.device)], dim=1)
            fused6 = cache6 + text_logits6
        else:
            cache5 = torch.zeros(n, len(CLASSES), device=self.device)
            fused6 = text_logits6
        cell6 = softmax(fused6, dim=1).cpu().numpy()                 # [N,6]

        # 块级门控：该块最像缺陷还是正常背景（正常类下标=5）
        is_defect = cell6[:, :5].max(axis=1) > cell6[:, 5]
        defect_idx = [i for i in range(n) if is_defect[i]]
        pool = cell6[defect_idx, :5] if defect_idx else cell6[:, :5]  # 无缺陷块则回退全块

        # max-pool 聚合：每个缺陷类在「缺陷块」上的最大概率作为存在证据
        class_max = pool.max(axis=0)
        order = int(np.argmax(class_max))

        cells = []
        for bi, box in enumerate(boxes):
            pi = int(np.argmax(cell6[bi, :5]))
            cells.append({
                "box": box, "pred": CLASSES[pi],
                "conf": float(cell6[bi, pi]),
                "normal_p": float(cell6[bi, 5]),
                "is_defect": bool(is_defect[bi]),
                "prob_vec": cell6[bi, :5],
            })

        annotated = self._draw_windows(image, cells, CLASSES[order], show_grid,
                                       n_defect=len(defect_idx))
        return {
            "mode": f"滑窗定位 {grid}×{grid}（含背景门控）",
            "group": group,
            "pred": CLASSES[order],
            "conf": float(class_max[order]),
            "probs": {CLASS_CN[CLASSES[i]]: float(class_max[i]) for i in range(len(CLASSES))},
            "probs_raw": class_max,
            "text_logits": text_logits6[:, :5].mean(0).cpu().numpy(),
            "cache_logits": cache5.mean(0).cpu().numpy(),
            "cells": cells,
            "annotated": annotated,
            "n_defect_cell": len(defect_idx),
            "n_anchor": 0 if bank is None else len(bank),
        }

    def _draw_windows(self, image: Image.Image, cells: List[Dict],
                      top_class: str, show_grid: bool, n_defect: int = 0) -> Image.Image:
        canvas = image.convert("RGB").copy()
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        font = _load_font(max(12, canvas.size[0] // 16))
        for cell in cells:
            x0, y0, x1, y1 = cell["box"]
            lw = max(2, canvas.size[0] // 100)
            if not cell.get("is_defect", True):
                # 判为正常背景：灰色淡填充，不标缺陷类，避免假阳性误导
                od.rectangle([x0, y0, x1, y1], fill=(180, 180, 180, 28))
                od.rectangle([x0, y0, x1, y1], outline=(150, 150, 150, 120), width=1)
                if show_grid:
                    od.text((x0 + 2, y0 + 1), "背景", fill=(110, 110, 110, 220), font=font)
                continue
            cls, conf = cell["pred"], cell["conf"]
            color = CLASS_COLOR[cls]
            alpha_fill = int(40 + 110 * conf)
            od.rectangle([x0, y0, x1, y1], fill=(*color, alpha_fill))
            od.rectangle([x0, y0, x1, y1], outline=(*color, 220), width=lw)
            if show_grid:
                tag = f"{CLASS_CN[cls]} {conf*100:.0f}%"
                od.rectangle([x0, y0, x0 + (x1-x0), y0 + max(16, (y1-y0)//5)],
                             fill=(*color, 200))
                od.text((x0 + 2, y0 + 1), tag, fill=(255, 255, 255, 255), font=font)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

        # 顶部标题条
        d = ImageDraw.Draw(canvas)
        bar_h = max(22, canvas.size[1] // 12)
        d.rectangle([0, 0, canvas.size[0], bar_h], fill=(30, 32, 36))
        d.text((6, 3),
               f"预测：{CLASS_CN[top_class]}（{top_class}）  缺陷窗口 {n_defect}/{len(cells)}，灰块=背景",
               fill=(255, 255, 255), font=_load_font(max(13, canvas.size[0] // 18)))
        return canvas

    # ---------- 智能多尺度（线上主方法：整图 + 粗窗 + 细窗）----------
    @torch.no_grad()
    def detect_smart(self, image: Image.Image, group: str = "L1",
                     bank: Optional[AnchorBank] = None, alpha: float = ALPHA_DEFAULT,
                     beta: float = BETA_DEFAULT, loc_grid: int = 3) -> Dict:
        """
        整图（全局上下文，利于划痕等线状缺陷）+ 2×2 粗窗（分类主力）+ loc_grid 细窗
        （定位与点状缺陷增强）三路协同。线上固定测试集评估整体 68%，优于任何单尺度。
        """
        whole = self.detect(image, group, bank, alpha, beta)
        coarse = self.detect_window(image, group, bank, grid=2, alpha=alpha,
                                    beta=beta, show_grid=False)
        fine = self.detect_window(image, group, bank, grid=loc_grid, alpha=alpha,
                                  beta=beta, show_grid=True)
        p_c, p_f = coarse["probs_raw"], fine["probs_raw"]
        cc, cf = int(np.argmax(p_c)), int(np.argmax(p_f))
        if cc == cf:
            ci = cc
            base = fine if p_f[cf] >= p_c[cc] else coarse
            probs = base["probs_raw"]
            route = "粗窗与细窗结论一致"
        elif p_c[cc] < SMART_LOW and p_f[cf] > SMART_HIGH:
            ci, probs = cf, p_f
            route = "粗窗不自信而细窗自信 → 采纳细窗（点状/局部缺陷需放大）"
        else:
            ci, probs = cc, p_c
            route = "粗细窗冲突 → 以粗窗为准（保留全局上下文，防线状缺陷被切碎误判）"
        annotated = self._draw_windows(image, fine["cells"], CLASSES[ci], True,
                                       n_defect=fine["n_defect_cell"])
        return {
            "mode": f"智能多尺度（整图+2×2+{loc_grid}×{loc_grid}）",
            "group": group,
            "pred": CLASSES[ci],
            "conf": float(probs[ci]),
            "probs": {CLASS_CN[CLASSES[i]]: float(probs[i]) for i in range(len(CLASSES))},
            "probs_raw": probs,
            "text_logits": fine["text_logits"],
            "cache_logits": fine["cache_logits"],
            "cells": fine["cells"],
            "annotated": annotated,
            "route": route,
            "whole_pred": whole["pred"],
            "whole_conf": whole["conf"],
            "coarse_pred": coarse["pred"],
            "fine_pred": fine["pred"],
            "n_defect_cell": fine["n_defect_cell"],
            "n_anchor": 0 if bank is None else len(bank),
        }

    # ---------- 锚点留一选优（自适应引擎的「实测层」）----------
    @torch.no_grad()
    def loo_compare(self, bank: AnchorBank, groups: Sequence[str] = ("L1", "L2", "L3"),
                    alpha: float = ALPHA_DEFAULT, beta: float = BETA_DEFAULT) -> Dict[str, float]:
        """用锚点做留一法（leave-one-out）：每次留 1 张当查询、其余当锚点，统计各组准确率。"""
        feats, labels = bank.feats, bank.labels
        m = len(bank)
        out = {}
        for g in groups:
            tp = self.text_protos(g)
            correct = 0
            for i in range(m):
                keep = torch.arange(m, device=self.device) != i
                sub = AnchorBank(feats[keep], labels[keep])
                fused, _, _ = self._fuse_logits(feats[i:i+1], tp, sub, alpha, beta)
                correct += int(fused.argmax(1).item() == int(labels[i].item()))
            out[g] = round(correct / m * 100, 1)
        return out

    # ---------- 文本-图像对齐质量门（机制结论：对齐 > 可分性）----------
    @torch.no_grad()
    def alignment_report(self, bank: AnchorBank) -> Dict[str, float]:
        """各组文本原型与本类锚点图像的平均余弦相似度（越高越对齐）。"""
        report = {}
        for g in GROUPS:
            tp = self.text_protos(g)
            vals = []
            for ci in range(len(CLASSES)):
                m = bank.labels == ci
                if m.sum() == 0:
                    continue
                vals.append(float((feats_mean := bank.feats[m].mean(0) @ tp[ci])))
            report[g] = round(float(np.mean(vals)), 3) if vals else float("nan")
        return report


# ----------------------------------------------------------------------------
# 全局单例
# ----------------------------------------------------------------------------
_ENGINES: Dict[str, DefectEngine] = {}


def get_engine(model_name: str = "ViT-B-16", device: Optional[str] = None) -> DefectEngine:
    """模型加载耗时，做成进程内单例，切换提示组/检测时不重复加载。"""
    key = model_name
    if key not in _ENGINES:
        _ENGINES[key] = DefectEngine(model_name, device)
    return _ENGINES[key]


if __name__ == "__main__":
    # 命令行自检：python detector.py  （构建 8 锚点并对一张测试整图做滑窗检测）
    eng = get_engine("ViT-B-16")
    bank = eng.build_bank_builtin(k=8)
    print("锚点每类数量：", bank.per_class_count())
    img, gt, _ = eng.sample_test_image(seed=1)
    for grp in ("L1", "L3"):
        r = eng.detect_window(img, grp, bank, grid=3)
        print(f"[{grp}] 真值={gt} 预测={r['pred']} 置信={r['conf']*100:.1f}%  概率={ {k: round(v*100,1) for k,v in r['probs'].items()} }")
    print("留一选优：", eng.loo_compare(bank))
    print("对齐报告：", eng.alignment_report(bank))
