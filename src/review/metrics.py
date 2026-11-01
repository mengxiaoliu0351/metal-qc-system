"""Operational counts and human-labelled quality are deliberately separate."""
from statistics import median
from math import ceil
from review.disposition import effective_disposition
from review.workflow import original_prediction


def operational_summary(records):
    classified = [r for r in records if r.get("disposition")]
    automatic = sum(bool(r["disposition"].get("auto_result")) for r in classified)
    pending = sum(effective_disposition(r)["requires_review"] for r in records)
    times = sorted(float(r["time_total"]) for r in classified if isinstance(r.get("time_total"), (int, float)) and r["time_total"] >= 0)
    human = [r for r in records if (r.get("human_verdict") or {}).get("confirmed") and (r.get('human_verdict') or {}).get('verdict') in ('ok', 'defect') and not r.get("review_reopened") and not r.get("needs_review")]
    comparison = [r for r in human if original_prediction(r) in ("normal", "defect")]
    fp = fn = tp = tn = 0
    review_seconds = []
    for r in comparison:
        truth = r["human_verdict"]["verdict"] == "defect"
        predicted = original_prediction(r) == "defect"
        tp += int(predicted and truth); tn += int(not predicted and not truth)
        fp += int(predicted and not truth); fn += int(not predicted and truth)
        seconds = r["human_verdict"].get("review_seconds")
        if isinstance(seconds, (int, float)) and 0 <= seconds <= 7200: review_seconds.append(seconds)
    n = len(classified)
    return {"new_policy_records": n, "automatic_decisions": automatic,
            "automatic_normal": sum(r["disposition"].get("auto_result") == "normal" for r in classified),
            "automatic_isolated": sum(r["disposition"].get("auto_result") == "defect" for r in classified),
            "automation_rate": automatic / n if n else None,
            "sent_to_review": sum(r["disposition"].get("requires_review", False) for r in classified),
            "human_review_rate": sum(r["disposition"].get("requires_review", False) for r in classified) / n if n else None,
            "pending_review": pending, "audit_selected": sum(r["disposition"].get("sampled", False) for r in classified),
            "median_seconds": median(times) if times else None,
            "p95_seconds": times[max(0, ceil(.95 * len(times)) - 1)] if times else None,
            "vlm_calls": sum(bool(r.get("vlm_called")) for r in classified),
            "vlm_call_rate": sum(bool(r.get("vlm_called")) for r in classified) / n if n else None,
            "reviewed_records": len(human), "labelled_comparisons": len(comparison),
            "reviewed_agreement": (tp + tn) / len(comparison) if comparison else None,
            "false_positive": fp, "false_negative": fn,
            "reviewed_recall": tp / (tp + fn) if tp + fn else None,
            "review_median_seconds": median(review_seconds) if review_seconds else None,
            "scope_note": "质效指标统计最近500条中的新策略记录；自动结果含抽查样本。人工一致性来自选择性复核，不能替代独立测试准确率。"}
