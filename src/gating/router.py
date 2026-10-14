# -*- coding: utf-8 -*-
"""
门控路由：异常分数 -> 快链路 / 慢链路
====================================
阈值 = 中位数 + 1.5×MAD = 0.2243（先导实验标定）
"""
THRESHOLD_DEFAULT = 0.2243


def gate(score: float, threshold: float = THRESHOLD_DEFAULT) -> dict:
    """根据异常分数路由到快/慢链路"""
    route = "fast" if score >= threshold else "slow"
    return {
        "route": route,
        "score": score,
        "threshold": threshold,
        "ratio": score / threshold if threshold else None,
        "label": ("高置信" if route == "fast" else "低置信"),
    }
