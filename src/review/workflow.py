"""Persist human truth, then train/calibrate candidates without changing live models.

The CLIP backbone and normal patch bank remain fixed. Reviewed images are grouped
by decoded pixel identity; benchmark validation/test images cannot enter feedback
training. Candidate publication and rollback are explicit, separately gated steps.
"""
import copy
import hashlib
import json
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def now():
    return datetime.now().isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_hash(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def benchmark_labels(data, paths, catalog=False):
    """Imported products use audited labels, not the legacy MVTec `good` folder."""
    if catalog:
        labels = np.asarray(data.get('labels', []))
        if labels.shape != (len(paths),) or not np.isin(labels, [0, 1]).all():
            raise ValueError('导入产品缺少有效的来源质量标签')
        return labels.astype(int)
    return np.array([int(Path(p).parent.name != 'good') for p in paths])


def original_prediction(record):
    automatic = (record.get("disposition") or {}).get("auto_result")
    if automatic in ("normal", "defect"):
        return automatic
    if record.get("disposition") and (record.get("gate") or {}).get("route") == "vlm_review":
        return None
    prediction = record.get("perception_prediction")
    if prediction in ("normal", "defect"):
        return prediction
    score, threshold = record.get("score"), (record.get("gate") or {}).get("threshold")
    if score is not None and threshold is not None:
        return "defect" if float(score) >= float(threshold) else "normal"
    return None


def measurement(labels, scores, low, high):
    y, s = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if not len(y):
        return None
    predicted = s >= low
    tn = int(np.sum((y == 0) & ~predicted)); fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted)); tp = int(np.sum((y == 1) & predicted))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    high_mask = s > high
    high_total = int(high_mask.sum())
    return {"n": len(y), "accuracy": (tn + tp) / len(y), "balanced_accuracy": balanced,
            "recall": recall, "false_positive_rate": fp / (tn + fp) if tn + fp else None,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "gray_rate": float(np.mean((s >= low) & (s <= high))),
            "high_risk_precision": float(y[high_mask].mean()) if high_total else None,
            "high_risk_false_positives": int(np.sum((y == 0) & high_mask)),
            "high_risk_count": high_total}


def calibrate(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores)
    if set(y.tolist()) != {0, 1} or not np.isfinite(s).all():
        raise ValueError("阈值校准需要有效的正常、缺陷两类样本")
    u = np.unique(s)
    choices = np.r_[np.nextafter(u[0], -np.inf), (u[:-1] + u[1:]) / 2, np.nextafter(u[-1], np.inf)]
    def rank(t):
        m = measurement(y, s, t, np.inf)
        return m["balanced_accuracy"], m["recall"], m["accuracy"], -m["false_positive_rate"]
    low = float(max(choices, key=rank))
    # High-risk local hints require at least three calibration positives and
    # >=98% precision, <=5% normal false positives on this calibration cohort.
    high = float(np.nextafter(max(float(s.max()), low), np.inf))
    for threshold in choices[choices > low]:
        m = measurement(y, s, low, float(threshold))
        fp_rate = m["high_risk_false_positives"] / int(np.sum(y == 0))
        if m["high_risk_count"] >= 3 and m["high_risk_precision"] >= .98 and fp_rate <= .05:
            high = float(threshold)
            break
    return low, high


class ReviewWorkflow:
    def __init__(self, root, results_dir, extractor=None):
        self.root, self.results = Path(root), Path(results_dir)
        self.history = self.results / "history"
        self.candidates = self.results / "feedback_candidates"
        self.runs = self.results / "model_iterations"
        self.config = self.results / "perception/banks/products.json"
        self.optimized = self.results / "perception/optimized"
        self.benchmark = self.results / "experiments/perception_upgrade_v1"
        self.extractor = extractor
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="review-iteration")
        self.futures = {}
        self._inspection_policy = None

    def policy(self):
        from review.disposition import InspectionPolicy
        if self._inspection_policy is None:
            self._inspection_policy = InspectionPolicy(self.root, self.results)
        return self._inspection_policy

    def records(self):
        for path in sorted(self.history.glob("*/*.json"), reverse=True):
            try:
                value = read(path)
                if isinstance(value, dict):
                    yield path, value
            except (OSError, ValueError):
                continue

    def locate(self, session_id="", day="", key=""):
        if day or key:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) or not key or any(c in key for c in '/\\:\x00') or key in (".", ".."):
                raise ValueError("检验记录标识无效")
            path = self.history / day / (key + ".json")
            if not path.is_file():
                raise FileNotFoundError("未找到检验记录")
            rec = read(path)
            if session_id and rec.get("session_id") != session_id:
                raise ValueError("记录与检测会话不一致")
            return path, rec
        matches = [(p, r) for p, r in self.records() if session_id and r.get("session_id") == session_id]
        if len(matches) != 1:
            raise FileNotFoundError("未找到唯一检测记录，请从复核列表重新打开")
        return matches[0]

    def queue(self, product="", status="all", offset=0, limit=50):
        from review.disposition import effective_disposition
        items = []
        summary = {"total": 0, "reviewed": 0, "correct": 0, "false_positive": 0, "false_negative": 0, "iteration_approved": 0}
        summary.update(pending=0, sampled=0, gray_pending=0, auto_normal=0, auto_isolated=0, legacy=0)
        for path, rec in self.records():
            if rec.get("record_scope") == "evaluation": continue
            if product and rec.get("product_key") != product:
                continue
            prediction = original_prediction(rec)
            annotation = rec.get("human_verdict") or {}
            disposition = effective_disposition(rec)
            confirmed = bool(annotation.get("confirmed")) and annotation.get("verdict") in ("ok", "defect") and not disposition["requires_review"]
            truth = ("normal" if annotation.get("verdict") == "ok" else "defect") if confirmed else None
            agreement = prediction == truth if confirmed and prediction else None
            summary["total"] += 1
            if disposition["requires_review"]: summary["pending"] += 1
            if disposition["requires_review"] and (rec.get("gate") or {}).get("route") == "vlm_review":
                summary["gray_pending"] += 1
            if disposition.get("sampled") and not confirmed: summary["sampled"] += 1
            if disposition["status"] == "auto_normal": summary["auto_normal"] += 1
            if disposition["status"] == "auto_isolated": summary["auto_isolated"] += 1
            if disposition["status"] == "legacy": summary["legacy"] += 1
            if confirmed:
                summary["reviewed"] += 1
                if annotation.get("use_for_iteration"): summary["iteration_approved"] += 1
                if agreement is True: summary["correct"] += 1
                if prediction == "defect" and truth == "normal": summary["false_positive"] += 1
                if prediction == "normal" and truth == "defect": summary["false_negative"] += 1
            if status == "pending" and not disposition["requires_review"] or status == "reviewed" and not confirmed or status == "errors" and agreement is not False:
                continue
            if status == "flagged" and (confirmed or not rec.get("needs_review")):
                continue
            if status == "sampled" and (not disposition.get("sampled") or confirmed): continue
            if status == "gray" and ((rec.get("gate") or {}).get("route") != "vlm_review" or not disposition["requires_review"]): continue
            if status == "automatic" and disposition["status"] not in ("auto_normal", "auto_isolated"): continue
            if status == "legacy" and disposition["status"] != "legacy": continue
            items.append({k: rec.get(k) for k in ("session_id", "timestamp", "sample_name", "product_key", "product_name", "score", "gate", "cognition", "human_verdict", "needs_review")})
            items[-1].update({"_day": path.parent.name, "_file": path.stem,
                              "system_prediction": prediction, "agreement": agreement, "disposition": disposition})
        summary["unreviewed"] = summary["total"] - summary["reviewed"]
        return {"ok": True, "summary": summary, "count": len(items), "records": items[offset:offset + limit]}

    def confirm(self, request):
        verdict = request.get("verdict")
        if verdict not in ("ok", "defect"):
            raise ValueError("verdict 必须为 ok 或 defect")
        with self.lock:
            path, rec = self.locate(request.get("session_id", ""), request.get("day", ""), request.get("key", ""))
            prediction = original_prediction(rec)
            approved = request.get("use_for_iteration") is True
            if approved and rec.get('iteration_eligible') is False:
                raise ValueError('该记录尚未绑定通过验证的质量模型；可保存最终核验，但不能回流已有产品模型')
            if approved and rec.get("record_scope") == "evaluation":
                raise ValueError("封存评估记录仅用于评价，不能批准回流训练或校准")
            source_label = (rec.get("source_annotation") or {}).get("label")
            if (approved and rec.get("record_scope") == "simulation" and source_label in ("normal", "defect")
                    and source_label != ("normal" if verdict == "ok" else "defect")):
                raise ValueError("核验演示标签与来源不一致，请先记录争议，暂不批准用于迭代")
            image = rec.get("input_png")
            if approved and (not image or not Path(image).is_file() or not rec.get("product_key")):
                raise ValueError("缺少持久化原图或产品，不能加入迭代样本")
            fingerprint = image_hash(image) if image and Path(image).is_file() else None
            if approved and fingerprint in self.reserved_images(rec.get('product_key')):
                raise ValueError('该图片已保留用于独立留出评估，可保存核验标签，但不能批准回流训练或校准')
            annotation = {"verdict": verdict, "defect_type": str(request.get("defect_type", ""))[:100] if verdict == "defect" else "",
                          "reviewer": str(request.get("reviewer", "未署名操作员")).strip()[:80] or "未署名操作员",
                          "note": str(request.get("note", ""))[:2000], "confirmed": True, "ts": now(),
                          "revision": uuid.uuid4().hex, "use_for_iteration": approved,
                          "system_prediction": prediction,
                          "agreement": prediction == ("normal" if verdict == "ok" else "defect") if prediction else None}
            annotation["verification_method"] = "dataset_label_replay" if rec.get("record_scope") == "simulation" else "operator_review"
            annotation["label_source"] = (rec.get("source_annotation") or {}).get("source", "operator_provided")
            seconds = request.get("review_seconds")
            if seconds is not None:
                import math
                if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 <= seconds <= 7200:
                    raise ValueError("复核耗时无效")
                annotation["review_seconds"] = round(seconds, 2)
            candidate = {"session_id": rec.get("session_id"), "product_key": rec.get("product_key"),
                         "input_png": image, "image_hash": fingerprint, "label": annotation,
                         "status": "approved" if approved else "pending_offline_review",
                         "record_day": path.parent.name, "record_key": path.stem}
            target = self.candidates / (path.parent.name + "_" + path.name)
            # An append-only revision copy makes later edits traceable.
            write(self.runs / "review_audit" / (annotation["revision"] + ".json"), candidate)
            write(target, candidate)
            rec["human_verdict"] = annotation
            rec["needs_review"] = False
            rec["review_reopened"] = False
            write(path, rec)
            suspended = self.policy().observe_review(rec, annotation)
            return {"ok": True, "learned": False, "candidate_saved": True, "annotation": annotation,
                    "automation_suspended": suspended,
                    "note": "已核验并发现自动结果误判：该产品当前版本自动处置已暂停，请校准后评估候选版本" if suspended else
                            "最终标签已保存；批准的样本可用于候选训练/校准，线上模型需评估后单独发布"}

    def approved(self, product):
        values = []
        reserved = self.reserved_images(product)
        for p in sorted(self.candidates.glob("*.json")):
            try:
                item = read(p)
                if item.get("product_key") == product and item.get("status") == "approved" and item.get("label", {}).get("confirmed"):
                    if item.get('image_hash') in reserved: continue
                    record_path = self.history / item['record_day'] / (item['record_key'] + '.json')
                    record = read(record_path)
                    if record.get('needs_review') or record.get('review_reopened'):
                        continue
                    if (record.get('human_verdict') or {}).get('revision') != item['label'].get('revision'):
                        continue
                    values.append(item)
            except (OSError, ValueError):
                continue
        return values

    def reserved_images(self, product):
        from review.data_identity import reserved_hashes
        return reserved_hashes(self.results)

    def review_digest(self, product):
        return digest(self.approved(product))

    def product(self, product):
        values = read(self.config)
        result = next((p for p in values if p["key"] == product), None)
        if result is None: raise ValueError("未知产品")
        return result

    def benchmark_folder(self, config):
        return self.results / config["review_benchmark"] if config.get("review_benchmark") else self.benchmark

    def run_path(self, run_id):
        if not re.fullmatch(r"[0-9]{8}T[0-9]{6}_[a-f0-9]{10}", run_id):
            raise ValueError("迭代任务标识无效")
        return self.runs / run_id

    def list_runs(self):
        current = {p["key"]: p.get("review_version") for p in read(self.config)}
        result = []
        for path in sorted(self.runs.glob("*/state.json"), reverse=True):
            item = read(path)
            item["active"] = current.get(item.get("product")) == item.get("id")
            item["has_publication"] = (path.parent / "publication.json").is_file()
            if item.get("status") in ("queued", "running") and item["id"] not in self.futures:
                item.update(status="interrupted", message="服务重启中断任务，请重新评估")
            result.append(item)
        return {"ok": True, "runs": result, "active_versions": current}

    def start(self, product, mode="model"):
        if mode not in ("model", "threshold"): raise ValueError("请选择模型重训或阈值校准")
        self.product(product)
        if not self.approved(product): raise ValueError("该产品尚无批准用于迭代的人工复核样本")
        with self.lock:
            if any(not f.done() for f in self.futures.values()): raise ValueError("已有迭代任务运行，请等待完成")
            run_id = datetime.now().strftime("%Y%m%dT%H%M%S_") + uuid.uuid4().hex[:10]
            write(self.run_path(run_id) / "state.json", {"id": run_id, "product": product, "mode": mode,
                  "status": "queued", "created": now(), "message": "等待处理"})
            self.futures[run_id] = self.executor.submit(self.execute, run_id)
        return {"ok": True, "id": run_id}

    def update(self, run_id, **changes):
        with self.lock:
            path = self.run_path(run_id) / "state.json"
            state = read(path); state.update(changes); write(path, state)

    def execute(self, run_id):
        try:
            self.update(run_id, status="running", message="读取冻结分区、校验样本身份")
            state = read(self.run_path(run_id) / "state.json")
            report = self.evaluate(run_id, state["product"], state["mode"])
            write(self.run_path(run_id) / "report.json", report)
            self.update(run_id, status="completed", finished=now(), report=report,
                        message="评估完成，可发布" if report["eligible_for_activation"] else "评估完成，保留当前模型；候选未达到发布条件")
        except Exception as exc:
            self.update(run_id, status="failed", finished=now(), message=str(exc))

    def evaluate(self, run_id, product, mode):
        from perception.feature_backends import binary_probability, export_binary
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from threadpoolctl import threadpool_limits
        config = copy.deepcopy(self.product(product))
        artifact_path = self.optimized / config["perception_artifact"]
        metadata = read(artifact_path)
        with np.load(artifact_path.parent / metadata["arrays"], allow_pickle=False) as z:
            current_params = {k: z[k].copy() for k in z.files}
        if metadata["product"] != product or not np.isclose(metadata["threshold"], config["threshold"], atol=1e-10, rtol=0):
            raise ValueError("当前模型与阈值不配套，请先修复配置")
        benchmark = self.benchmark_folder(config)
        split_path = benchmark / (product + "_split.json")
        cache_path = benchmark / (product + "_features.npz")
        split = read(split_path)
        with np.load(cache_path, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        # Frozen manifest and cache were generated together by optimize_perception.py.
        if str(data["signature"]) != hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest():
            raise ValueError("冻结分区与特征缓存不匹配")
        # Verify the original cache signature above, then normalize paths for Linux.
        split = {key: [p.replace("\\", "/") for p in value] if isinstance(value, list) else value
                 for key, value in split.items()}
        paths = [p.replace("\\", "/") for p in data["paths"].tolist()]
        if len(set(paths)) != len(paths): raise ValueError("特征缓存含重复路径")
        indices = {p: i for i, p in enumerate(paths)}
        train = [indices[p] for p in split["train"]]
        validation = [indices[p] for p in split["validation"]]
        test = [indices[p] for p in split["test"]]
        if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
            raise ValueError("冻结训练、校准、测试分区相互重叠")
        legacy = {p.relative_to(self.root).as_posix() for p in sorted((self.root / "data/mvtec" / product / "train/good").glob("*.png"))[:80]}
        validation += [indices[p] for p in split["normal_calibration"] if p not in legacy]
        labels = benchmark_labels(data, paths, catalog=bool(config.get('catalog_manifest')))
        identities = {}
        for i, path in enumerate(paths):
            fingerprint = image_hash(self.root / path)
            partition = "train" if i in train else "calibration" if i in validation else "audit" if i in test else "excluded"
            if fingerprint in identities and identities[fingerprint][1] != partition and partition != "excluded" and identities[fingerprint][1] != "excluded":
                raise ValueError("冻结分区存在内容相同的图像，需先修复训练/校准/测试隔离")
            identities[fingerprint] = (i, partition)
        normal_hashes = {image_hash(self.root / p): i for i, p in enumerate(split["normal_memory"])}
        if any(h in identities and identities[h][1] in ("calibration", "audit") for h in normal_hashes):
            raise ValueError("正常训练参考图与冻结校准/测试图内容重叠")
        if config.get("catalog_manifest"):
            for sample in read(self.results / config["catalog_manifest"])["samples"]:
                if sample["role"] == "test":
                    identities[sample["pixel_hash"]] = (-1, "sealed_test")
        for fingerprint in self.reserved_images(product):
            identities[fingerprint] = (-1, 'reserved_evaluation')
        xt = np.concatenate([data["global_features"][train], data["normal_global"]]).copy()
        yt = np.r_[labels[train], np.zeros(len(data["normal_global"]), dtype=int)]
        train_slots = {p: i for i, p in enumerate(train)}
        approved = self.approved(product)
        review_signature = digest(approved)
        grouped = {}
        excluded = []; usage = []
        for item in approved:
            image = item.get("input_png")
            if not image or not Path(image).is_file():
                excluded.append({"session_id": item.get("session_id"), "reason": "原图缺失"}); continue
            fingerprint = image_hash(image)
            if item.get("image_hash") and item["image_hash"] != fingerprint:
                excluded.append({"session_id": item.get("session_id"), "reason": "复核后原图已改变"}); continue
            grouped.setdefault(fingerprint, []).append(item)
        extra = {"train": [], "calibration": [], "audit": []}
        for fingerprint, group in grouped.items():
            truths = {int(x["label"]["verdict"] == "defect") for x in group}
            if len(truths) != 1:
                excluded.append({"session_id": group[0].get("session_id"), "reason": "同图存在冲突的人工标签"}); continue
            item, truth = group[-1], truths.pop()
            known = identities.get(fingerprint)
            if known and known[1] != "train":
                excluded.append({"session_id": item.get("session_id"), "reason": "冻结校准/测试或参考库重叠样本，不加入反馈训练与调参"}); continue
            if fingerprint in normal_hashes:
                slot = len(train) + normal_hashes[fingerprint]
                yt[slot] = truth; role = "train"
                if known and known[0] in train_slots: yt[train_slots[known[0]]] = truth
            elif known:
                yt[train_slots[known[0]]] = truth; role = "train"
            else:
                bucket = int(hashlib.sha256((product + fingerprint).encode()).hexdigest()[:8], 16) % 10
                role = "train" if bucket < 6 else "calibration" if bucket < 8 else "audit"
                extra[role].append({**item, "truth": truth})
            usage.append({"session_id": item.get("session_id"), "role": role, "image_hash": fingerprint})
        counts = {k: sum(x["role"] == k for x in usage) for k in extra}
        if mode == "threshold" and not counts["calibration"]:
            raise ValueError("尚无独立新增校准样本；可继续积累复核，或选择模型重训。冻结训练样本不能拿来调阈值")
        if mode == "model" and not counts["train"]:
            raise ValueError("尚无可用的复核训练样本；冻结校准/测试图片不会回流训练")
        self.update(run_id, message="提取新增样本特征，准备训练与校准")
        extra_data = {}
        for role, items in extra.items():
            if items:
                if self.extractor is None: raise ValueError("未配置新增图片特征提取器")
                features, baseline = self.extractor(product, [x["input_png"] for x in items])
                features, baseline = np.asarray(features), np.asarray(baseline)
                if features.shape != (len(items), xt.shape[1]) or baseline.shape != (len(items),) or not np.isfinite(features).all() or not np.isfinite(baseline).all():
                    raise ValueError("新增图片特征无效")
                extra_data[role] = {"global_features": features, "clip_scores": baseline,
                                    "y": np.array([x["truth"] for x in items])}
        if "train" in extra_data:
            xt = np.concatenate([xt, extra_data["train"]["global_features"]])
            yt = np.r_[yt, extra_data["train"]["y"]]
        # StandardScaler preserves float32 inputs, while the portable exported
        # model computes in float64. Fit in float64 to preserve exact parity.
        xt = np.asarray(xt, dtype=np.float64)
        params = {k: v.copy() for k, v in current_params.items()}
        proposed_mode = metadata["mode"]
        cv_results = None
        if mode == "model":
            self.update(run_id, message="训练监督判别头；主干与正常记忆库保持固定")
            nfold = min(5, int(np.bincount(yt.astype(int), minlength=2).min()))
            if nfold < 2: raise ValueError("训练集中每类至少需要 2 张图像")
            def classifier(c):
                return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=20261023))])
            with threadpool_limits(limits=4):
                cv = StratifiedKFold(nfold, shuffle=True, random_state=20261023)
                cv_results = {c: float(cross_val_score(classifier(c), xt, yt, cv=cv, scoring="balanced_accuracy").mean()) for c in (.01, .1, 1.)}
                fitted = classifier(max(cv_results, key=cv_results.get)).fit(xt, yt)
            params = export_binary(fitted, "global"); proposed_mode = "global_binary"
            if not np.allclose(binary_probability(xt[:8], params, "global"), fitted.predict_proba(xt[:8])[:, 1], rtol=1e-10, atol=1e-12):
                raise ValueError("导出模型与训练模型推理不一致")
        def predict(values, parameters, score_mode):
            if score_mode == "normal_memory": return values["clip_scores"]
            probability = binary_probability(values["global_features"], parameters, "global")
            if score_mode == "global_binary": return probability
            if score_mode == "fast_fusion": return binary_probability(np.c_[values["clip_scores"], probability], parameters, "fusion")
            raise ValueError("当前迭代工具支持监督二分类/快速融合模型，请先检查产品模式")
        old_scores, new_scores = predict(data, current_params, metadata["mode"]), predict(data, params, proposed_mode)
        old_cal, new_cal, ycal = old_scores[validation], new_scores[validation], labels[validation]
        if "calibration" in extra_data:
            extra_cal = extra_data["calibration"]
            old_cal = np.r_[old_cal, predict(extra_cal, current_params, metadata["mode"])]
            new_cal = np.r_[new_cal, predict(extra_cal, params, proposed_mode)]
            ycal = np.r_[ycal, extra_cal["y"]]
        low, high = calibrate(ycal, new_cal)
        current_low, current_high = float(config["threshold"]), float(config["t_high"])
        old_val = measurement(ycal, old_cal, current_low, current_high)
        new_val = measurement(ycal, new_cal, low, high)
        old_test = measurement(labels[test], old_scores[test], current_low, current_high)
        new_test = measurement(labels[test], new_scores[test], low, high)
        audit = None
        if "audit" in extra_data:
            a = extra_data["audit"]
            audit = {"current": measurement(a["y"], predict(a, current_params, metadata["mode"]), current_low, current_high),
                     "candidate": measurement(a["y"], predict(a, params, proposed_mode), low, high)}
        reasons = []
        metric_names = {"accuracy": "准确率", "balanced_accuracy": "平衡准确率", "recall": "缺陷召回率"}
        for name, old, new in [("校准集", old_val, new_val), ("冻结回归集", old_test, new_test)]:
            for metric in ("accuracy", "balanced_accuracy", "recall"):
                if new[metric] is None or old[metric] is None or new[metric] + 1e-12 < old[metric]: reasons.append(f"{name}的{metric_names[metric]}未通过不回退检查")
        if audit:
            for metric in ("accuracy", "balanced_accuracy", "recall"):
                old, new = audit["current"][metric], audit["candidate"][metric]
                if old is not None and new is not None and new + 1e-12 < old:
                    reasons.append("新增审计样本的" + metric_names[metric] + "回退")
        improved = any(new_val[k] > old_val[k] + 1e-12 for k in ("accuracy", "balanced_accuracy", "recall")) or (new_val["gray_rate"] < old_val["gray_rate"] - 1e-12 and new_val["high_risk_false_positives"] <= old_val["high_risk_false_positives"])
        if not improved: reasons.append("尚未观察到校准指标或复核调用量的改善")
        directory = self.run_path(run_id)
        np.savez_compressed(directory / "candidate.npz", **params)
        new_meta = {**metadata, "mode": proposed_mode, "name": "CLIP 复核样本再训练二分类" if mode == "model" else metadata["name"] + " · 复核阈值校准",
                    "arrays": "candidate.npz", "threshold": low, "t_high": high, "review_version": run_id}
        write(directory / "candidate.json", new_meta)
        return {"id": run_id, "product": product, "mode": mode, "base_config": config,
                "base_config_digest": digest(config), "review_digest": review_signature,
                "base_artifact_sha256": file_hash(artifact_path.parent / metadata["arrays"]),
                "benchmark_sha256": file_hash(cache_path), "split_sha256": file_hash(split_path),
                "candidate_sha256": file_hash(directory / "candidate.npz"),
                "candidate_metadata_sha256": file_hash(directory / "candidate.json"),
                "thresholds": {"current": [current_low, current_high], "candidate": [low, high]},
                "validation": {"current": old_val, "candidate": new_val},
                "regression": {"current": old_test, "candidate": new_test}, "new_audit": audit,
                "feedback_usage": counts, "used_samples": usage, "excluded_samples": excluded,
                "training_n": len(yt), "cv_training_only": cv_results,
                "eligible_for_activation": not reasons, "gate_reasons": reasons,
                "scope": "固定CLIP主干；更新监督判别头/门控阈值；正常patch记忆库不回灌",
                "protocol": "同图按RGB像素去重；冻结校准/测试样本不回流；新图按内容哈希固定60/20/20分组。校准集选择阈值，冻结公开测试分区仅作回归门槛，已非新产线盲测。",
                "high_threshold_rule": "校准样本上高风险区至少3件、精确率≥98%、正常误报≤5%；样本有限，不能视为产线保证"}

    def activate(self, run_id):
        with self.lock:
            directory = self.run_path(run_id)
            report = read(directory / "report.json")
            if not report.get("eligible_for_activation"): raise ValueError("候选未通过发布门槛，不能应用")
            product = report["product"]
            if digest(self.product(product)) != report["base_config_digest"]: raise ValueError("线上配置已变化，请基于最新版本重新评估")
            if self.review_digest(product) != report["review_digest"]: raise ValueError("复核标签已变化，请重新评估")
            used_hashes = {item["image_hash"] for item in report["used_samples"]}
            for item in self.approved(product):
                if item.get("image_hash") in used_hashes:
                    path = Path(item["input_png"])
                    if not path.is_file() or image_hash(path) != item["image_hash"]:
                        raise ValueError("用于评估的复核图像已变化，请重新评估")
            current_meta = read(self.optimized / report["base_config"]["perception_artifact"])
            checks = [(directory / "candidate.npz", report["candidate_sha256"]),
                      (directory / "candidate.json", report["candidate_metadata_sha256"]),
                      (self.optimized / current_meta["arrays"], report["base_artifact_sha256"]),
                      (self.benchmark_folder(report["base_config"]) / (product + "_features.npz"), report["benchmark_sha256"]),
                      (self.benchmark_folder(report["base_config"]) / (product + "_split.json"), report["split_sha256"])]
            if any(file_hash(p) != expected for p, expected in checks): raise ValueError("评估后模型或数据发生变化，请重新评估")
            name = "review_" + product + "_" + run_id
            import shutil
            self.optimized.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / "candidate.npz", self.optimized / (name + ".npz"))
            meta = read(directory / "candidate.json"); meta["arrays"] = name + ".npz"
            write(self.optimized / (name + ".json"), meta)
            configs = read(self.config)
            position = next(i for i, p in enumerate(configs) if p["key"] == product)
            before = copy.deepcopy(configs[position])
            configs[position].update(perception_artifact=name + ".json", perception_name=meta["name"],
                                    threshold=meta["threshold"], t_high=meta["t_high"], review_version=run_id,
                                    score_kind="binary_model_score",
                                    calibration={"method": "human-reviewed-feedback-calibration", "report": str(directory / "report.json"),
                                                 "high_threshold_method": report["high_threshold_rule"]})
            write(directory / "publication.json", {"product": product, "before": before, "after": configs[position], "published": now(),
                  "before_arrays_sha256": report["base_artifact_sha256"],
                  "before_metadata_sha256": file_hash(self.optimized / before["perception_artifact"])})
            write(self.config, configs)
            return {"ok": True, "active_version": run_id, "note": "候选模型与阈值已成套发布，正常记忆库未修改"}

    def rollback(self, run_id):
        with self.lock:
            publication = read(self.run_path(run_id) / "publication.json")
            configs = read(self.config)
            position = next(i for i, p in enumerate(configs) if p["key"] == publication["product"])
            if digest(configs[position]) != digest(publication["after"]):
                raise ValueError("仅能回滚当前仍在使用的发布版本，不能覆盖后续更新")
            previous_path = self.optimized / publication["before"]["perception_artifact"]
            previous = read(previous_path)
            if file_hash(previous_path) != publication["before_metadata_sha256"] or file_hash(self.optimized / previous["arrays"]) != publication["before_arrays_sha256"]:
                raise ValueError("发布前的模型文件已改变，不能直接回滚")
            configs[position] = publication["before"]
            write(self.config, configs)
            publication["rolled_back"] = now()
            write(self.run_path(run_id) / "publication.json", publication)
            return {"ok": True, "note": "已恢复该产品发布前的模型与阈值"}
