# -*- coding: utf-8 -*-
"""
adaptive.py —— 自适应提示策略引擎（研究结论 → 系统决策）
================================================================
双层决策，正好对应论文的两条主线：
  1) 先验层（统计规则）：把「六条实验结论」写成确定性规则，锚点极少也能用；
  2) 选优层（锚点留一实测）：锚点够时，用 leave-one-out 在 L1/L2/L3 里实测选优，
     从而自动兼容 pitted（知识有益）、ViT-L-14（L2 反超）等反例，而不是写死 L1。
  3) 质量门：用「文本-图像对齐」而非「文本可分性」判断提示是否靠谱（阈值 0.216）。

所有阈值均来自真实实验（B-16 10 seeds / L-14 3 seeds），不是拍脑袋。
"""
from typing import Dict, List, Optional, Tuple

from detector import GROUPS, ALIGN_THRESHOLD

# 交叉点：每类锚点达到该数量后，提示组差异收敛、L3 不再有害（full=60 时 L3 93.7 微超 L1 91.3）
CROSSOVER_K = 60
# 低锚点区上限：K<=8 是机制词损害最显著的区间（K=8 L1 80.2 vs L3 60.6，p<0.001）
LOW_ANCHOR_K = 9
# LOO 实测领先多少才覆盖先验（避免被留一噪声带偏）
LOO_MARGIN = 3.0
# 锚点达到多少，LOO 才相对可信（太少方差大）
LOO_MIN_K = 3

# CLIP 零样本先验弱、必须靠锚点的细粒度低纹理类（K0 L1 仅 0~13%）
HARD_ZEROSHOT = {"crazing", "patches", "inclusion"}


def prior_rule(k: int, model_name: str = "ViT-B-16") -> Tuple[str, List[str], List[str]]:
    """先验层：依据锚点规模返回 (推荐组, 理由, 警告)。"""
    reasons, warnings = [], []
    if k <= 0:
        group = "L1"
        reasons.append("当前为零样本（无锚点）：现象级视觉描述显著优于裸类名与通用模板"
                       "（ViT-L-14 零样本 L1 44.7% vs L3 30.0%）。")
        warnings.append("零样本下发纹/斑块/夹杂等细粒度类几乎不可识（发纹仅 3.3%），"
                        "请在「建模」页为每类添加锚点，不要指望提示词救活。")
    elif k < LOW_ANCHOR_K:
        group = "L1"
        reasons.append(f"处于低锚点少样本区（每类 {k} 张，<{LOW_ANCHOR_K}）：机制级知识词会显著"
                       "损害匹配（K=8 时 L1 80.2% vs L3 60.6%，差 19.6 点，10 seeds 全部成立，"
                       "p<0.001），应只用现象级视觉描述。")
        reasons.append("此区间锚点数是最强增益来源（纯锚点 K=8 即达 84%），"
                       "优先补锚点比打磨提示措辞更有效。")
    elif k < CROSSOVER_K:
        group = "L1"
        reasons.append(f"每类 {k} 张锚点：现象描述仍最稳；L1/L2 可由锚点留一实测决定"
                       "（ViT-L-14 上该区间 L2 曾小幅反超，故交给实测层）。")
        warnings.append("机制词组 L3 在锚点充足前仍不建议使用。")
    else:
        group = "L1"
        reasons.append(f"每类 {k}≥{CROSSOVER_K} 张，已越过收益交叉点：提示组差异收敛"
                       "（full 时 L1/L2/L3 为 91.3/91.3/93.7），可按留一实测自由选择。")
    return group, reasons, warnings


def _quality_gate(align: Optional[Dict[str, float]], chosen: str) -> List[str]:
    notes = []
    if not align:
        return notes
    val = align.get(chosen)
    if val is None:
        return notes
    if val < ALIGN_THRESHOLD:
        notes.append(f"质量门：{chosen} 的文本-图像对齐度 {val:.3f} 低于阈值 "
                     f"{ALIGN_THRESHOLD:.3f}（裸类名水平），提示与缺陷图对不齐，建议更换模板。")
    base_a = align.get("Baseline-A")
    if base_a is not None and val - base_a >= 0.02:
        notes.append(f"质量门通过：{chosen} 对齐度 {val:.3f} 明显高于裸类名 "
                     f"{base_a:.3f}，验证「对齐>可分性」。")
    return notes


def recommend(k: int, loo: Optional[Dict[str, float]] = None,
              align: Optional[Dict[str, float]] = None,
              model_name: str = "ViT-B-16") -> Dict:
    """
    综合决策。返回：
      chosen            最终推荐组
      prior_group       先验层推荐
      loo_group         留一实测最优（可能为 None）
      reasons/warnings  可直接展示给用户的中文依据
      decision_path     决策路径说明（答辩用，体现不是写死）
    """
    prior, reasons, warnings = prior_rule(k, model_name)
    chosen = prior
    loo_group, loo_note = None, None

    # 选优层：锚点够才信 LOO
    if loo and k >= LOO_MIN_K:
        loo_group = max(loo, key=lambda g: loo[g])
        best_score, prior_score = loo[loo_group], loo.get(prior)
        if prior_score is not None and best_score - prior_score >= LOO_MARGIN:
            chosen = loo_group
            loo_note = (f"留一实测：{loo_group} 在你当前锚点上达 {best_score:.1f}%，"
                        f"比先验推荐的 {prior}（{prior_score:.1f}%）高 {best_score-prior_score:.1f} 点，"
                        "采用实测结果（可自动兼容「知识对个别类有益」这类反例）。")
        else:
            loo_note = (f"留一实测最优为 {loo_group}（{best_score:.1f}%），与先验 {prior}"
                        f"（{prior_score if prior_score is not None else '-'}%）差距小于 "
                        f"{LOO_MARGIN:.0f} 点，维持先验以保证稳健。")
        reasons.append(loo_note)

    # 交叉点之前，实测层也不允许把结果翻到 L3（除非已越过交叉点）
    if k < CROSSOVER_K and chosen == "L3" and loo_group == "L3":
        chosen = prior if prior != "L3" else "L1"
        warnings.append("虽然留一实测偏高，但锚点未越过交叉点，机制词组 L3 的统计风险仍在，回退现象描述。")

    reasons += _quality_gate(align, chosen)

    path = ("零/少样本先验规则" if k < LOO_MIN_K or not loo else
            "先验规则 → 锚点留一实测 → 对齐质量门")
    return {
        "chosen": chosen,
        "prior_group": prior,
        "loo_group": loo_group,
        "reasons": reasons,
        "warnings": warnings,
        "decision_path": path,
        "loo": loo or {},
        "align": align or {},
        "k": k,
    }


def detect_guard(group: str, k: int, pred_class: Optional[str] = None) -> List[str]:
    """检测时的实时提示（用户手动选了某个组时，结合结论给出风险提示）。"""
    tips = []
    if group == "L3" and 0 < k < CROSSOVER_K:
        tips.append(f"当前每类仅 {k} 个锚点，实验证明此区间机制词组 L3 平均掉点约 20 个"
                    "（对发纹类最严重，可差 79 点）。如结果异常请切回 L1。")
    if group == "Baseline-A":
        tips.append("裸类名与缺陷图像对齐度最低（0.216），少样本下接近随机，仅作对照。")
    if k == 0 and pred_class in HARD_ZEROSHOT:
        tips.append(f"{pred_class} 属于零样本弱先验类，本次判定可信度有限，建议加锚点后复检。")
    return tips


if __name__ == "__main__":
    # 规则自检
    for k in (0, 2, 8, 30, 60):
        r = recommend(k, loo={"L1": 70.0, "L2": 66.0, "L3": 55.0})
        print(f"K={k:>2} -> {r['chosen']:<11} | {r['decision_path']}")
