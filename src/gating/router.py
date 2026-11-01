# -*- coding: utf-8 -*-
"""
三区间门控路由
==============
异常分数 s 由当前产品的感知模型给出（s 越大表示异常证据越强）。
- s < t_low          : 本地低风险，不调 VLM，直接"未发现明显异常"
- t_low <= s <= t_high: 灰区，调 VLM 深度复核 + 国标 RAG
- s > t_high         : 本地高风险，不调 VLM，直接"明显异常，建议人工复核"

原型 t_low 可采用正常样本 median + 1.5*MAD；优化模型使用验证集阈值。
t_high 控制灰区复核范围，工程取值不能视为独立验证的准确率保证。
阈值与模型成套加载，不同产品和模型的分数不能直接互换。
最终判定仍由人工确认签发。
"""


def gate(score: float, t_low: float, t_high: float = None) -> dict:
    """三区间路由。t_high 缺省时取 t_low 的 1.5 倍（保守灰区）。"""
    if t_high is None or t_high <= t_low:
        t_high = t_low * 1.5
    if score < t_low:
        route = "low_risk"
        label = "低风险 · 本地快检"
    elif score <= t_high:
        route = "vlm_review"
        label = "灰区 · VLM深度复核"
    else:
        route = "high_risk"
        label = "高风险 · 本地快检"
    return {
        "route": route,
        "score": score,
        "t_low": t_low,
        "t_high": round(t_high, 4),
        "label": label,
    }
