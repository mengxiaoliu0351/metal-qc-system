# -*- coding: utf-8 -*-
"""
v4 提示词库（中文 EI 期刊投稿版）
课题：面向金属缺陷的分层领域知识提示少样本识别

三变量分离设计：
  L1 纯现象  ->  L2 现象+成因  ->  L3 现象+成因+机制风险（知识梯度）
  L0a 语义无关、等长；L0b 语义在目标、无知识、等长（对照）
  基线 A 短标签 / B 通用模板 / Normal 类别扩展

规则：
  1) 所有 L3 增量均为"纯机制词"（stress concentration / fatigue life 等），
     不含可视觉对应的词（如 crack、fracture），保证"知识 vs 视觉描述"归因干净。
  2) L0a / L0b 与 L3 需 token 级等长（用 token_length_report 校准到 ±2）。
"""
from typing import Dict, List

# NEU-DET 使用的 5 类（v4 主动移除 rolled-in scale，见方案文件）
CLASSES: List[str] = ["scratches", "patches", "crazing", "pitted_surface", "inclusion"]

# ---------- 主干实验组：领域知识梯度 ----------
# L1 纯视觉现象
L1: Dict[str, str] = {
    "scratches":      "a photo of steel surface with scratch defect",
    "patches":        "a photo of steel surface with patch defect",
    "crazing":        "a photo of steel surface with crazing defect",
    "pitted_surface": "a photo of steel surface with pitting defect",
    "inclusion":      "a photo of steel surface with inclusion defect",
}

# L2 现象 + 轧制成因
L2: Dict[str, str] = {
    "scratches":      "a photo of steel surface scratch, which is caused by roll wear or foreign particle abrasion during steel rolling process",
    "patches":        "a photo of steel surface patch, which is caused by uneven cooling or surface contamination during steel rolling process",
    "crazing":        "a photo of steel surface crazing, which is caused by excessive rolling strain and rapid thermal cycling during steel rolling process",
    "pitted_surface": "a photo of steel surface pitting, which is caused by oxide scale indentation or water spot corrosion during steel rolling process",
    "inclusion":      "a photo of steel surface inclusion, which is caused by non-metallic impurities embedded in the steel slab during casting",
}

# L3 现象 + 成因 + 机制风险（增量全为纯机制词）
L3: Dict[str, str] = {
    "scratches":      "a photo of steel surface scratch induced by rolling abrasion, which degrades surface integrity, raises local stress concentration and deteriorates downstream stamping and forming quality",
    "patches":        "a photo of steel surface patch caused by uneven cooling, which disrupts surface homogeneity, creates local property variation and affects downstream painting and coating quality",
    "crazing":        "a photo of steel surface crazing caused by excessive rolling strain, which weakens surface cohesion, increases local stress concentration and reduces the service fatigue life",
    "pitted_surface": "a photo of steel surface pitting caused by oxide scale indentation, which reduces surface smoothness, creates stress concentration points and shortens the component fatigue life",
    "inclusion":      "a photo of steel surface inclusion caused by casting impurities, which breaks material continuity, creates internal stress concentration and substantially shortens the component fatigue life",
}

# L3-V 视觉化知识（存档，探索性）：与 L3 等长、增量全为视觉外观描述。
# 结论：零样本有效（37 vs L3 的 29），但少样本未达预期（与 L3 相当），未纳入论文。
L3V: Dict[str, str] = {  # noqa: F841（存档代码，未被主流程引用）
    "scratches":      "a photo of steel surface scratch defect which appears as thin sharp contrasting lines of varying lengths crossing the otherwise smooth bright metal surface in many different directions",
    "patches":        "a photo of steel surface patch defect which appears as large very dark regions with sharply defined clear edges standing out clearly against the lighter surrounding metal surface",
    "crazing":        "a photo of steel surface crazing defect which appears as a fine dense grainy rough textured surface with many tiny intersecting hairlines spread densely across it",
    "pitted_surface": "a photo of steel surface pitting defect which appears as small sparse dark round spots of varying sizes scattered unevenly across a bright smooth background surface",
    "inclusion":      "a photo of steel surface inclusion defect which appears as a large smooth dark wide elongated region softly blended into the surrounding lighter surface and running along the surface",
}

# ---------- 等长控制组（类无关，~25 词）----------
# 注：以下两段已用 CLIP BPE tokenizer 校准到与 L3（31 token）相差 <=2。
# 修改后必须重跑 token_length_report 复校（目标 |L0x - L3| <= 2）。
L0a: str = ("the smooth steel plate has a quite uniform and glossy appearance under stable "
            "industrial lighting conditions, while the polished metal surface clearly and "
            "entirely reflects the surrounding workshop environment")
L0b: str = ("a photo of steel surface with a defect, the image is collected by an "
            "industrial camera inside the workshop with a standard shooting angle and "
            "a clean, even background")

# ---------- 基线对照组 ----------
BASELINE_A: Dict[str, str] = {c: c for c in CLASSES}  # 短单词标签
BASELINE_B: str = "a photo of steel surface defect"     # 通用模板

# ---------- 类别扩展组（原"负提示"，正常类）----------
NORMAL_WEAK: str = "a photo of steel surface with normal surface"
NORMAL_STRONG: str = "defect-free steel, high-quality, passed inspection"


def build_prompt_sets() -> Dict[str, Dict[str, List[str]]]:
    """返回 {prompt_group: {class_name: [prompt, ...]}}。
    组名：L1 / L2 / L3 / L0a / L0b / Baseline-A / Baseline-B / Normal。
    L0a/L0b/Baseline-B 为类无关文本，对每个类重复同一句；
    Normal 组每个类两个提示（弱/强），供类别扩展实验使用。
    """
    sets: Dict[str, Dict[str, List[str]]] = {}
    sets["L1"] = {c: [L1[c]] for c in CLASSES}
    sets["L2"] = {c: [L2[c]] for c in CLASSES}
    sets["L3"] = {c: [L3[c]] for c in CLASSES}
    # 注：L3V（视觉化知识）为探索性，few-shot 未达预期，未纳入论文主实验（见 08 报告附录）
    sets["L0a"] = {c: [L0a] for c in CLASSES}
    sets["L0b"] = {c: [L0b] for c in CLASSES}
    sets["Baseline-A"] = {c: [BASELINE_A[c]] for c in CLASSES}
    sets["Baseline-B"] = {c: [BASELINE_B] for c in CLASSES}
    sets["Normal"] = {c: [NORMAL_WEAK, NORMAL_STRONG] for c in CLASSES}
    return sets


def token_length_report(tokenizer) -> Dict[str, int]:
    """打印各提示组的 token 数，用于校准等长控制组。
    用法：
        import open_clip
        _, _, tokenizer = open_clip.create_model_and_transforms("ViT-B-16")
        token_length_report(tokenizer)
    目标：L0a、L0b 与 L3 的 token 数相差 <= 2。
    """
    sets = build_prompt_sets()
    report = {}
    print(f"{'group':<14}{'tokens':>6}   prompt (以 scratches 类为例)")
    for group in ["L1", "L2", "L3", "L0a", "L0b", "Baseline-A", "Baseline-B", "Normal"]:
        prompt = sets[group]["scratches"][0]
        n = len(tokenizer.encode(prompt))
        report[group] = n
        print(f"{group:<14}{n:>6}   {prompt[:80]}...")
    return report


if __name__ == "__main__":
    print("运行 token 校准：")
    print("  python -c \"import open_clip; from prompts import token_length_report; "
          "_,_,tok=open_clip.create_model_and_transforms('ViT-B-16'); token_length_report(tok)\"")
