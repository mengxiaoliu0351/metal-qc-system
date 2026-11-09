"""Operational counts and human-labelled quality are deliberately separate."""
from statistics import median
from math import ceil
from review.disposition import effective_disposition
from review.workflow import original_prediction


def operational_summary(records):
    records = [r for r in records if r.get("record_scope", "production") == "production"]
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


def gray_decision_summary(records):
    """Report gray-zone auto decisions and confirmed errors without treating reviews as a blind test."""
    fields = ("total", "auto_normal", "auto_defect", "uncertain", "reviewed_auto_normal",
              "reviewed_auto_defect", "normal_misses", "defect_false_positives", "reviewed_uncertain")
    summary = {field: 0 for field in fields}
    summary.update(products={}, scope_note="仅统计灰区新规则记录并按产品、图像内容、模型版本去重。已复核样本受抽查与例外选择影响；误判数不能当作独立测试准确率。自动结果只作软件归档或隔离。")
    seen = set()
    for rec in records:
        if rec.get("record_scope", "production") != "production": continue
        disposition = rec.get("disposition") or {}
        assessment = disposition.get("gray_assessment") or {}
        if (rec.get("gate") or {}).get("route") != "vlm_review" or assessment.get("version") != "gray-decision-v1":
            continue
        product = rec.get("product_key") or "unknown"
        identity = rec.get("image_hash") or rec.get("session_id")
        key = (product, identity, disposition.get("model_key"))
        if identity and key in seen:
            continue
        if identity:
            seen.add(key)
        annotation = rec.get("human_verdict") or {}
        reviewed = (annotation.get("confirmed") is True and annotation.get("verdict") in ("ok", "defect")
                    and not rec.get("review_reopened") and not rec.get("needs_review"))
        proposal = disposition.get("auto_result") if disposition.get("auto_result") in ("normal", "defect") else "abstain"
        bucket = summary["products"].setdefault(product, {field: 0 for field in fields})
        for target in (summary, bucket):
            target["total"] += 1
            target[{"normal": "auto_normal", "defect": "auto_defect"}.get(proposal, "uncertain")] += 1
            if reviewed and proposal == "normal":
                target["reviewed_auto_normal"] += 1
                target["normal_misses"] += int(annotation["verdict"] == "defect")
            elif reviewed and proposal == "defect":
                target["reviewed_auto_defect"] += 1
                target["defect_false_positives"] += int(annotation["verdict"] == "ok")
            elif reviewed:
                target["reviewed_uncertain"] += 1
    return summary


def audit_window_summary(records, period_key):
    """Count automatic outcomes and their review tickets for the active audit period."""
    result = {"period_key": period_key, "automatic": 0, "selected": 0,
              "selected_pending": 0, "selected_reviewed": 0, "confirmed_errors": 0,
              "by_route": {"low_risk": 0, "vlm_review": 0, "high_risk": 0}}
    for rec in records:
        if rec.get("record_scope", "production") != "production": continue
        action = rec.get("disposition") or {}
        if action.get("audit_period_key") != period_key or action.get("auto_result") not in ("normal", "defect"):
            continue
        result["automatic"] += 1
        route = (rec.get("gate") or {}).get("route")
        if route in result["by_route"]:
            result["by_route"][route] += 1
        if action.get("sampled"):
            result["selected"] += 1
            human = rec.get("human_verdict") or {}
            reviewed = (human.get("confirmed") is True and human.get("verdict") in ("ok", "defect")
                        and not rec.get("review_reopened") and not rec.get("needs_review"))
            result["selected_reviewed" if reviewed else "selected_pending"] += 1
            if reviewed:
                truth = "normal" if human["verdict"] == "ok" else "defect"
                result["confirmed_errors"] += int(truth != action["auto_result"])
    result["observed_rate"] = result["selected"] / result["automatic"] if result["automatic"] else None
    return result
