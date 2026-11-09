"""Selective inspection for the application prototype; machine actions are never labels.

Routing thresholds still decide VLM use. Separate calibration-only guard bands,
scope checks and frozen regression gates decide whether to archive/isolate or ask
for review. No physical line release, rejection or training is performed here.
"""
import hashlib
import json
import math
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
from PIL import Image

from review.workflow import read, write, file_hash, now


LABELS = {
    "auto_normal": "系统正常 · 自动归档", "auto_isolated": "疑似缺陷 · 自动隔离",
    "needs_review": "待人工复核", "sampled_review": "抽查待核验",
    "human_confirmed": "人工已确认", "legacy": "历史记录 · 未作业分流",
}


def image_quality(image):
    gray = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32)
    reasons = []
    if min(image.size) < 128: reasons.append("图像分辨率不足")
    if float(gray.std()) < 5: reasons.append("图像近乎空白或对比度不足")
    if float(np.mean(gray < 3)) > .99: reasons.append("图像几乎全黑")
    if float(np.mean(gray > 252)) > .99: reasons.append("图像几乎全白")
    return {"ok": not reasons, "reasons": reasons, "width": image.width, "height": image.height,
            "contrast": round(float(gray.std()), 2)}


def validate_references(cognition, references):
    result = dict(cognition or {})
    ids = result.get("ref_ids", [])
    allowed = {str(r.get("id")) for r in references if r.get("id")}
    invalid = not isinstance(ids, list) or any(not isinstance(x, str) or x not in allowed for x in ids)
    result["citation_check"] = {"ok": not invalid, "invalid_ids": ids if invalid else [],
                                "has_citations": bool(ids) if isinstance(ids, list) else False}
    return result


def gray_decision_evidence(cognition, score, low, high):
    """A deterministic abstaining rule; VLM prose alone never authorizes an archive."""
    cog = cognition or {}
    conclusion = cog.get("vlm_conclusion") or cog.get("conclusion")
    base = {"version": "gray-decision-v1", "proposal": "abstain", "vlm_conclusion": conclusion}
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not low < high or not low <= score <= high:
        return {**base, "reason": "灰区分数或阈值无效"}
    position = (score - low) / (high - low)
    base["gray_position"] = round(position, 4)
    if not cog.get("ok"):
        return {**base, "reason": "视觉分析未完成"}
    citation = cog.get("citation_check") or {}
    if citation.get("ok") is not True or citation.get("has_citations") is not True:
        return {**base, "reason": "知识材料引用缺失或编号校验未通过"}
    if not str(cog.get("defect_desc") or "").strip():
        return {**base, "reason": "图像描述不足"}
    if cog.get("review_version") == "focus-v2":
        expected = {"visible_defect": "不合格", "no_visible_defect": "合格"}
        assessment = cog.get("visual_assessment")
        if (cog.get("image_adequacy") != "adequate" or not isinstance(assessment, str)
                or expected.get(assessment) != conclusion
                or not isinstance(cog.get("defect_location"), str)
                or not cog["defect_location"].strip()):
            return {**base, "reason": "外观证据不明确、图像不足或结构化结论冲突"}
    if conclusion in ("正常", "合格") and cog.get("defect_type") == "正常" and position <= .25:
        return {**base, "proposal": "normal", "reason": "灰区靠近低风险端，视觉分析明确为正常且引用可核对"}
    known_defects = {"划痕", "弯曲", "色差", "裂纹", "凸起", "凹陷"}
    if conclusion == "不合格" and cog.get("defect_type") in known_defects and position >= .5:
        return {**base, "proposal": "defect", "reason": "灰区靠近高风险端，视觉分析指出具体缺陷且引用可核对"}
    return {**base, "reason": "分数位置与视觉分析未同时满足自动分类条件"}


def effective_disposition(record):
    original = record.get("disposition") or {}
    human = record.get("human_verdict") or {}
    confirmed = human.get("confirmed") and human.get("verdict") in ("ok", "defect")
    if record.get("needs_review") or record.get("review_reopened"):
        return {**original, "status": "needs_review", "label": "反馈标记 · 待复核", "requires_review": True,
                "reason": "使用者标记了结果问题，请核验真实标签", "sampled": original.get("sampled", False)}
    if confirmed:
        return {**original, "status": "human_confirmed", "label": LABELS["human_confirmed"],
                "requires_review": False, "final_action": "normal_archive" if human["verdict"] == "ok" else "isolate"}
    if original:
        return original
    # Never retroactively label old scores as automatically accepted or rejected.
    return {"status": "legacy", "label": LABELS["legacy"], "requires_review": False,
            "reason": "历史记录未执行当前作业策略，可主动打开核验", "sampled": False}


def _unit(array):
    a = np.asarray(array, dtype=np.float64)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


class InspectionPolicy:
    def __init__(self, root, results):
        self.root, self.results = Path(root), Path(results)
        self.path = self.results / "inspection_policy.json"
        self.lock = threading.RLock()
        self.cache = {}
        self.centroids = None

    def settings(self):
        if self.path.exists(): return read(self.path)
        return {"version": 2, "mode": "assisted", "sample_rate": .10, "audit_period": "day",
                "holds": {}, "revision": "initial"}

    def configure(self, request):
        mode, rate = request.get("mode"), request.get("sample_rate")
        period = request.get("audit_period", self.settings().get("audit_period", "day"))
        if mode != "assisted" or isinstance(rate, bool) or not isinstance(rate, (int, float)) or not .05 <= rate <= 1:
            raise ValueError("仅支持辅助作业，抽查比例须为5%至100%")
        if period not in ("day", "week"):
            raise ValueError("抽查周期须为每日或每周")
        operator = str(request.get("operator", "")).strip()[:80]
        if not operator: raise ValueError("请填写策略调整人员")
        with self.lock:
            config = self.settings()
            before = {k: config.get(k) for k in ("mode", "sample_rate", "audit_period")}
            config.update(mode=mode, sample_rate=float(rate), audit_period=period,
                          revision=now(), operator=operator)
            write(self.path, config)
            import uuid
            write(self.results / "policy_audit" / (uuid.uuid4().hex + ".json"),
                  {"before": before, "after": {k: config[k] for k in ("mode", "sample_rate", "audit_period")}, "operator": operator, "ts": now()})
        return self.public_settings()

    def public_settings(self):
        config = self.settings()
        return {"ok": True, **{k: config[k] for k in ("mode", "sample_rate", "revision", "holds")},
                "audit_period": config.get("audit_period", "day"),
                "scope": "公开数据集原型的系统归档/隔离策略；自动结果不构成人工标签或产线认证",
                "gray_policy": "灰区仅在分数位置、结构化AI结论和引用编号同时满足规则时自动归档；其他情况转人工。此为原型规则，需持续抽查与独立验证"}

    @staticmethod
    def period_key(period, timestamp=None):
        when = datetime.fromisoformat(timestamp) if timestamp else datetime.now(ZoneInfo("Asia/Shanghai"))
        if period == "week":
            year, week, _ = when.isocalendar()
            return f"{year}-W{week:02d}"
        return when.strftime("%Y-%m-%d")

    def profile(self, config):
        from perception.feature_backends import binary_probability
        product = config["key"]
        if config.get("catalog_key"):
            from products.qualification import catalog_profile
            profile = catalog_profile(self.results, config)
            write(self.results / "automation_profiles" / (product + ".json"), profile)
            return profile
        meta_path = self.results / "perception/optimized" / config["perception_artifact"]
        meta = read(meta_path)
        arrays = meta_path.parent / meta["arrays"]
        split_path = self.results / "experiments/perception_upgrade_v1" / (product + "_split.json")
        cache_path = split_path.with_name(product + "_features.npz")
        exclusions_path = Path(__file__).with_name('calibration_exclusions.json')
        stamp = (json.dumps(config, sort_keys=True), *(str(p) + str(p.stat().st_mtime_ns) + str(p.stat().st_size)
                   for p in (meta_path, arrays, split_path, cache_path, exclusions_path)))
        if stamp in self.cache: return self.cache[stamp]
        if meta["product"] != product or not np.isclose(meta["threshold"], config["threshold"], atol=1e-10, rtol=0):
            raise ValueError("模型与产品阈值未配套")
        split = read(split_path)
        with np.load(cache_path, allow_pickle=False) as z: data = {k: z[k].copy() for k in z.files}
        if str(data["signature"]) != hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest():
            raise ValueError("冻结特征与分区不匹配")
        paths = [str(p).replace("\\", "/") for p in data["paths"]]
        index = {p: i for i, p in enumerate(paths)}
        validation = [index[p.replace("\\", "/")] for p in split["validation"]]
        exclusion = read(exclusions_path)[product]
        if exclusion['split_signature'] != str(data['signature']):
            raise ValueError('校准排除清单与冻结分区不匹配')
        memory = set(exclusion['legacy_memory'])
        calibration = [index[p.replace('\\', '/')] for p in split['normal_calibration'] if p.replace('\\', '/') not in memory]
        if meta.get('normal_calibration_count') != len(calibration):
            raise ValueError('校准样本数与模型配套记录不一致')
        validation += calibration
        test = [index[p.replace("\\", "/")] for p in split["test"]]
        if set(validation) & set(test): raise ValueError("校准与回归分区重叠")
        with np.load(arrays, allow_pickle=False) as z: params = {k: z[k].copy() for k in z.files}
        global_scores = binary_probability(data["global_features"], params, "global")
        if meta["mode"] == "global_binary": scores = global_scores
        elif meta["mode"] == "fast_fusion": scores = binary_probability(np.c_[data["clip_scores"], global_scores], params, "fusion")
        else: raise ValueError("该模型类型尚未验证自动处置")
        y = np.array([int(Path(p).parent.name != "good") for p in paths])
        vs, vy = scores[validation], y[validation]
        if set(vy.tolist()) != {0, 1}: raise ValueError("自动处置校准缺少两类样本")
        # Fixed conservative margin: choose from calibration, never tune on test.
        auto_low = min(float(config["threshold"]), .5 * float(vs[vy == 1].min()))
        auto_high = max(float(config["t_high"]), float(np.nextafter(vs[vy == 0].max(), np.inf)))
        ts, ty = scores[test], y[test]
        low, high = ts < auto_low, ts > auto_high
        metrics = {"n": len(test), "auto_normal": int(low.sum()), "auto_isolated": int(high.sum()),
                   "auto_normal_misses": int(np.sum(low & (ty == 1))),
                   "auto_isolation_false_positives": int(np.sum(high & (ty == 0)))}
        centroid = _unit(data["normal_global"].mean(axis=0))
        val_similarity = _unit(data["global_features"][validation]) @ centroid
        cutoff = float(max(-1, np.quantile(val_similarity, .01) - .03))
        model_key = hashlib.sha256((json.dumps(config, sort_keys=True) + file_hash(meta_path) + file_hash(arrays)).encode()).hexdigest()
        profile = {"product": product, "model_key": model_key, "auto_low": auto_low, "auto_high": auto_high,
                   "allow_normal": metrics["auto_normal"] >= 3 and metrics["auto_normal_misses"] == 0,
                   "allow_isolate": metrics["auto_isolated"] >= 3 and metrics["auto_isolation_false_positives"] == 0,
                   "scope_min_similarity": cutoff, "centroid": centroid.tolist(), "regression": metrics,
                   "calibration_n": len(validation), "scope": "公开数据集原型；冻结分区重复用于回归门槛，非新产线盲测",
                   "rule": "校准正例最低分的一半与原下阈值取较小值；隔离线取原上阈值与校准负例最高分较大值；回归分区仅作否决门槛"}
        self.cache = {stamp: profile, **dict(list(self.cache.items())[-5:])}
        write(self.results / "automation_profiles" / (product + ".json"), profile)
        return profile

    def scope_check(self, config, vector, profile):
        if vector is None: return {"ok": False, "reason": "缺少产品适用性特征"}
        v = _unit(np.asarray(vector).reshape(-1))
        if len(v) != len(profile["centroid"]) or not np.isfinite(v).all():
            return {"ok": False, "reason": "产品适用性特征无效"}
        similarity = float(v @ np.asarray(profile["centroid"]))
        if self.centroids is None:
            self.centroids = {}
            for path in (self.results / "experiments/perception_upgrade_v1").glob("*_features.npz"):
                with np.load(path, allow_pickle=False) as z:
                    self.centroids[path.name.removesuffix("_features.npz")] = _unit(z["normal_global"].mean(axis=0))
        self.centroids[config["key"]] = _unit(profile["centroid"])
        ranked = sorted(((float(v @ c), p) for p, c in self.centroids.items() if len(c) == len(v)), reverse=True)
        mismatch = bool(ranked and ranked[0][1] != config["key"] and ranked[0][0] - similarity > .025)
        ok = similarity >= profile["scope_min_similarity"] and not mismatch
        return {"ok": bool(ok), "similarity": round(similarity, 4), "min_similarity": round(profile["scope_min_similarity"], 4),
                "closest_product": ranked[0][1] if ranked else config["key"],
                "reason": "适用性初筛通过" if ok else "图像可能不属于当前产品或采集分布，请核实产品与图像"}

    def decide(self, record, config=None, vector=None):
        settings = self.settings()
        result = {"policy_version": "selective-v1", "mode": settings["mode"], "sample_rate": settings["sample_rate"],
                  "policy_revision": settings["revision"], "sampled": False, "model_key": None,
                  "auto_result": None, "system_action": "hold_for_review", "scope": "application_prototype"}
        def pending(reason, code):
            return {**result, "status": "needs_review", "label": LABELS["needs_review"], "requires_review": True,
                    "reason": reason, "reason_code": code}
        if not (record.get("quality_check") or {}).get("ok", True): return pending("图像质量初筛未通过，请重新采集或人工核验", "input_quality")
        if record.get('gate', {}).get('route') == 'processing_error': return pending('检测处理异常，请重试或人工核验', 'processing_error')
        if not config or record.get("gate", {}).get("route") == "general": return pending("未建库产品仅提供AI线索，需要人工核验", "unregistered_product")
        try:
            profile = self.profile(config)
            result["model_key"] = profile["model_key"]
            result["auto_thresholds"] = [profile["auto_low"], profile["auto_high"]]
            scope = self.scope_check(config, vector, profile)
            result["scope_check"] = scope
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return pending("当前产品版本缺少配套自动处置评估，请先验证", "unvalidated_model")
        if not scope["ok"]: return pending(scope["reason"], "scope_mismatch")
        hold = settings["holds"].get(config["key"] + ":" + profile["model_key"])
        if hold: return pending("核验或封存评估发现该版本误判，自动处置已暂停，请完成校准与候选评估", "quality_hold")
        route = record.get("gate", {}).get("route")
        cog = record.get("cognition") or {}
        score = record.get("score")
        if route == "vlm_review":
            evidence = gray_decision_evidence(cog, score, float(config["threshold"]), float(config["t_high"]))
            result["gray_assessment"] = evidence
            if cog.get("review_version") == "focus-v2":
                return pending("新版外观分析尚未通过独立验证，已保留观察证据，请人工核验", "unvalidated_visual_review")
            if evidence["proposal"] == "normal":
                if not profile["allow_normal"]: return pending("该产品自动正常门槛未通过，请人工核验", "unvalidated_normal")
                result.update(status="auto_normal", auto_result="normal", system_action="normal_archive",
                              decision_source="gray_vlm_rag", reason="灰区深度分析满足自动正常归档规则；保留AI证据和抽查入口")
            elif evidence["proposal"] == "defect":
                if not profile["allow_isolate"]: return pending("该产品自动隔离门槛未通过，请人工核验", "unvalidated_isolate")
                result.update(status="auto_isolated", auto_result="defect", system_action="isolate",
                              decision_source="gray_vlm_rag", reason="灰区深度分析满足疑似缺陷隔离规则；不执行物理报废")
            else:
                code = "vlm_failed" if not cog.get("ok") else (
                    "invalid_citation" if (cog.get("citation_check") or {}).get("ok") is not True else "gray_uncertain")
                return pending("灰区深度分析仍不确定：" + evidence["reason"] + "；请人工核验", code)
        else:
            if not isinstance(score, (int, float)) or not math.isfinite(score): return pending("缺少有效感知分数", "invalid_score")
            if route == "low_risk" and score < profile["auto_low"] and profile["allow_normal"]:
                result.update(status="auto_normal", auto_result="normal", system_action="normal_archive",
                              decision_source="local", reason="低风险且通过适用性与自动归档门槛，已保存系统正常判断")
            elif route == "high_risk" and score > profile["auto_high"] and profile["allow_isolate"]:
                result.update(status="auto_isolated", auto_result="defect", system_action="isolate",
                              decision_source="local", reason="高风险且通过隔离门槛，已加入疑似缺陷隔离记录；不执行报废")
            else: return pending("靠近自动处置边界或该分区回归未达标，请人工核验", "guard_band")
        # Stable content-based sampling within the selected audit window.
        period = settings.get("audit_period", "day")
        period_key = self.period_key(period, record.get("timestamp"))
        token = hashlib.sha256((record["image_hash"] + config["key"] + profile["model_key"] + period_key + "audit-v2").encode()).hexdigest()
        sampled = int(token[:16], 16) / 2**64 < settings["sample_rate"]
        result.update(requires_review=sampled, sampled=sampled, initial_status=result["status"],
                      decision_reason=result["reason"], sample_ticket=token[:16],
                      audit_period=period, audit_period_key=period_key)
        if sampled: result.update(status="sampled_review", reason="自动处理结果被稳定随机抽查选中，请核验真实标签")
        result["label"] = LABELS[result["status"]]
        return result

    def observe_evaluation(self, batch):
        """Publisher truth may veto a release, never train or relax its thresholds."""
        if batch.get("purpose") != "evaluation" or batch.get("status") not in ("completed", "completed_with_errors"):
            raise ValueError("仅接受已完成的封存评估批次")
        config = batch["recipe"]["config"]
        profile = self.profile(config)
        errors = []
        for item in batch["items"]:
            if item["status"] != "completed" or item.get("dataset_truth") not in ("normal", "defect"):
                continue
            action = (item.get("result") or {}).get("disposition") or {}
            if action.get("auto_result") and action["auto_result"] != item["dataset_truth"]:
                if action.get("model_key") != profile["model_key"]:
                    raise ValueError("自动结果与封存模型版本不匹配")
                errors.append(item["id"])
        result = {"batch_id": batch["id"], "product": config["key"], "model_key": profile["model_key"],
                  "source": "publisher_label_sealed_test", "automatic_error_count": len(errors),
                  "error_items": errors, "automatic_release_vetoed": bool(errors), "ts": now(),
                  "scope": "来源标签封存评估；只否决自动处置，不用测试集调参，不构成人工核验"}
        with self.lock:
            write(self.results / "release_evaluations" / (batch["id"] + ".json"), result)
            if errors:
                settings = self.settings()
                settings["holds"][config["key"] + ":" + profile["model_key"]] = {
                    **result, "reason": "封存评估观察到自动判断错误，暂停该版本自动处置"}
                settings["revision"] = now()
                write(self.path, settings)
        return result

    def observe_review(self, record, annotation):
        d = record.get("disposition") or {}
        if not d.get("auto_result") or not d.get("model_key"): return False
        truth = "normal" if annotation["verdict"] == "ok" else "defect"
        if d["auto_result"] == truth: return False
        with self.lock:
            config = self.settings()
            config["holds"][record["product_key"] + ":" + d["model_key"]] = {
                "product": record["product_key"], "session_id": record.get("session_id"), "ts": now(),
                "reason": "人工核验发现自动结果与真实标签不一致", "reviewer": annotation["reviewer"]}
            write(self.path, config)
        return True
