"""Persistent product drafts, disjoint samples and explicitly published builds."""
import io
import json
import re
import shutil
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from review.workflow import digest, file_hash, image_hash, measurement, now, read, write

ROLES = ("bank", "train", "calibration", "qualification", "test", "feedback")


class ProductCatalog:
    def __init__(self, root, results, extractor=None, lock=None):
        self.root, self.results = Path(root), Path(results)
        self.directory = self.results / "product_catalog"
        self.registry = self.results / "perception/banks/products.json"
        self.extractor = extractor
        self.lock = lock or threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="product-build")
        self.futures = {}

    def folder(self, key):
        if not re.fullmatch(r"product_[a-f0-9]{12}", key):
            raise ValueError("产品标识无效")
        return self.directory / key

    def create(self, request):
        values = {}
        for key, limit in (("name", 80), ("sku", 100), ("view", 80), ("source", 500), ("license", 200)):
            value = request.get(key, "")
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ValueError("请完整填写产品名称、型号、视图、数据来源及使用许可")
            values[key] = value.strip()
        with self.lock:
            if len(list(self.directory.glob("*/product.json"))) >= 20:
                raise ValueError("当前工作区最多接入20个新产品")
            key = "product_" + uuid.uuid4().hex[:12]
            product = {"key": key, **values, "created": now(), "revision": uuid.uuid4().hex,
                       "samples": [], "active_build": None}
            write(self.folder(key) / "product.json", product)
        return self.detail(key)

    def detail(self, key):
        product = read(self.folder(key) / "product.json")
        result = {k: v for k, v in product.items() if k != "samples"}
        counts = Counter((s["role"], s["label"]) for s in product["samples"])
        result["counts"] = {r: {label: counts[r, label] for label in ("normal", "defect")} for r in ROLES}
        result["total"] = len(product["samples"])
        result["builds"] = []
        for path in sorted((self.folder(key) / "builds").glob("*/state.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
            state = read(path)
            if state["status"] in ("queued", "running") and state["id"] not in self.futures:
                state.update(status="interrupted", message="服务重启中断建库，请重新启动")
            result["builds"].append(state)
        return result

    def list(self):
        return [self.detail(p.parent.name) for p in sorted(self.directory.glob("*/product.json"))]

    def validate_identities(self, product):
        from review.data_identity import aliases, canonical, reserved_hashes
        mapping, reserved = aliases(self.results), reserved_hashes(self.results)
        seen = set()
        for sample in product['samples']:
            if sample['role'] != 'test' and sample['pixel_hash'] in reserved:
                raise ValueError('图片属于已封存评估数据或其缩放／压缩副本，只能留作测试，不能用于建库、训练、校准或资格检查')
            identity = canonical(sample['pixel_hash'], mapping)
            if identity in seen:
                raise ValueError('同一来源图像的不同副本重复入库，请先去重并保持分区一致')
            seen.add(identity)

    def add_image(self, key, content, filename, role, label, group_id=""):
        if role not in ROLES or label not in ("normal", "defect"):
            raise ValueError("样本用途或标签无效")
        if role == "bank" and label != "normal":
            raise ValueError("正常记忆库仅接收确认正常的样本")
        if not content or len(content) > 10 * 1024 * 1024:
            raise ValueError("每张图片须小于10MB")
        with Image.open(io.BytesIO(content)) as source:
            if source.width * source.height > 20_000_000 or min(source.size) < 128:
                raise ValueError("图片尺寸须不少于128像素且不超过2000万像素")
            if source.format not in ("PNG", "JPEG", "BMP", "WEBP"):
                raise ValueError("请使用PNG、JPEG、BMP或WEBP图片")
            image = source.convert("RGB")
        import hashlib
        fingerprint = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        if encoded.tell() > 10 * 1024 * 1024:
            raise ValueError("规范化后的图片大于10MB，请降低分辨率后导入")
        group_id = str(group_id).strip()[:160]
        with self.lock:
            product = read(self.folder(key) / "product.json")
            from review.data_identity import aliases, canonical, reserved_hashes
            mapping = aliases(self.results)
            identity = canonical(fingerprint, mapping)
            if role != 'test' and fingerprint in reserved_hashes(self.results):
                raise ValueError('图片属于已封存评估数据或其缩放／压缩副本，只能留作测试，不能用于建库、训练、校准或资格检查')
            for sample in product["samples"]:
                if canonical(sample["pixel_hash"], mapping) == identity:
                    if sample["role"] != role or sample["label"] != label:
                        raise ValueError("同图或其已确认副本已在其他分区或标签中，禁止重复跨区使用")
                    return {"ok": True, "duplicate": True, "total": len(product["samples"])}
                if group_id and sample.get("group_id") == group_id and sample["role"] != role:
                    raise ValueError("同一实物或来源组必须保持在同一分区")
            if len(product["samples"]) >= 6000:
                raise ValueError("每产品最多6000张样本")
            storage = sum(s["storage_bytes"] if "storage_bytes" in s else (self.root / s["path"]).stat().st_size for s in product["samples"])
            if storage + encoded.tell() > 2 * 1024 ** 3 or shutil.disk_usage(self.results).free < 2 * 1024 ** 3:
                raise ValueError("产品样本存储已达2GB或剩余磁盘不足2GB，请整理数据后继续")
            path = self.folder(key) / "images" / role / ("good" if label == "normal" else "defect") / (fingerprint + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded.getbuffer())
            product["samples"].append({"path": path.relative_to(self.root).as_posix(), "pixel_hash": fingerprint,
                "role": role, "label": label, "original_name": Path(str(filename)).name[:160], "group_id": group_id,
                "label_source": product["source"], "imported": now()})
            product["samples"][-1]["storage_bytes"] = encoded.tell()
            product["revision"] = uuid.uuid4().hex
            write(self.folder(key) / "product.json", product)
        return {"ok": True, "duplicate": False, "total": len(product["samples"])}

    def start(self, key, mode="normal_memory"):
        if mode not in ("normal_memory", "global_binary"):
            raise ValueError("请选择正常记忆库或有标签整图判别")
        with self.lock:
            if any(not f.done() for f in self.futures.values()):
                raise ValueError("已有产品正在建库，请等待完成")
            product = read(self.folder(key) / "product.json")
            self.validate_identities(product)
            if sum(s["role"] == "bank" for s in product["samples"]) < 10:
                raise ValueError("至少导入10张独立正常建库图")
            if sum(s["role"] == "bank" for s in product["samples"]) > 200:
                raise ValueError("当前CPU工作站每产品最多使用200张正常建库图")
            if sum(s["role"] == "calibration" and s["label"] == "normal" for s in product["samples"]) < 10:
                raise ValueError("至少导入10张独立正常校准图")
            if self.extractor is None:
                raise ValueError("未配置产品特征提取器")
            build = uuid.uuid4().hex
            folder = self.folder(key) / "builds" / build
            write(folder / "input.json", product)
            write(folder / "state.json", {"id": build, "product": key, "mode": mode, "status": "queued",
                                           "created": now(), "message": "等待建库", "revision": product["revision"]})
            self.futures[build] = self.executor.submit(self.execute, key, build)
        return {"ok": True, "id": build}

    def update(self, folder, **values):
        with self.lock:
            state = read(folder / "state.json")
            state.update(**values)
            write(folder / "state.json", state)

    def execute(self, key, build):
        folder = self.folder(key) / "builds" / build
        try:
            self.update(folder, status="running", message="提取正常样本及校准特征")
            product, state = read(folder / "input.json"), read(folder / "state.json")
            self.validate_identities(product)
            # Sealed test images never participate in fitting, threshold selection or qualification.
            samples = [s for s in product["samples"] if s["role"] not in ("test", "feedback")]
            for sample in samples:
                if image_hash(self.root / sample["path"]) != sample["pixel_hash"]:
                    raise ValueError("导入图片已变化，请重新导入并建库")
            data = self.extractor(samples, lambda done, total: self.update(folder, message=f"提取特征 {done}/{total}"))
            report, config = self.finish_build(folder, product, samples, data, state["mode"])
            write(folder / "candidate_config.json", config)
            write(folder / "report.json", report)
            self.update(folder, status="completed", finished=now(), message="建库完成，可查看验证结果后接入", report=report)
        except Exception as exc:
            self.update(folder, status="failed", finished=now(), message=str(exc))

    def finish_build(self, folder, product, samples, data, mode):
        from perception.feature_backends import binary_probability, export_binary
        from review.workflow import calibrate
        indices = {role: [i for i, s in enumerate(samples) if s["role"] == role] for role in ROLES}
        labels = np.array([s["label"] == "defect" for s in samples], dtype=int)
        global_features, raw = np.asarray(data["global_features"]), np.asarray(data["clip_scores"])
        if len(raw) != len(samples) or global_features.ndim != 2 or len(global_features) != len(samples):
            raise ValueError("提取的特征与样本清单不匹配")
        if not np.isfinite(global_features).all() or not np.isfinite(raw).all():
            raise ValueError("特征包含无效数值")
        params = {}
        scores = raw
        if mode == "global_binary":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            train = indices["train"]
            if any(np.sum(labels[train] == label) < 2 for label in (0, 1)):
                raise ValueError("整图判别训练每类至少需要2张独立训练图")
            model = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=.1, class_weight="balanced", max_iter=2000))])
            model.fit(global_features[train].astype(np.float64), labels[train])
            params = export_binary(model, "global")
            scores = binary_probability(global_features, params, "global")
        calibration = indices["calibration"]
        ys, ss = labels[calibration], scores[calibration]
        if set(ys.tolist()) == {0, 1}:
            low, high = calibrate(ys, ss)
        else:
            low = float(np.quantile(ss, .99))
            high = float(max(low * 1.5, np.nextafter(low, np.inf)))
        bank_name = f"{product['key']}_{folder.name}.npy"
        banks = self.results / "perception/banks"
        banks.mkdir(parents=True, exist_ok=True)
        np.save(banks / bank_name, data["bank"])
        artifact_name = f"{product['key']}_{folder.name}.json"
        optimized = self.results / "perception/optimized"
        optimized.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(optimized / artifact_name.replace(".json", ".npz"), **params)
        name = "正常样本记忆库" if mode == "normal_memory" else "CLIP 有标签整图判别"
        write(optimized / artifact_name, {"product": product["key"], "mode": mode, "name": name,
              "arrays": artifact_name.replace(".json", ".npz"), "threshold": low, "t_high": high})
        split = {"train": [samples[i]["path"] for i in indices["train"]],
                 "validation": [samples[i]["path"] for i in calibration], "normal_calibration": [],
                 "normal_memory": [samples[i]["path"] for i in indices["bank"]],
                 "test": [samples[i]["path"] for i in indices["qualification"]]}
        write(folder / (product["key"] + "_split.json"), split)
        np.savez_compressed(folder / (product["key"] + "_features.npz"),
            signature=np.array(digest(split)), paths=np.array([s["path"] for s in samples]),
            global_features=global_features, clip_scores=raw,
            normal_global=global_features[indices["bank"]], labels=labels)
        config = {"key": product["key"], "name": product["name"] + " · " + product["view"],
                  "sku": product["sku"], "view": product["view"], "source": product["source"],
                  "bank": bank_name, "threshold": low, "t_high": high, "perception_artifact": artifact_name,
                  "perception_name": name, "review_version": "onboard_" + folder.name,
                  "review_benchmark": folder.relative_to(self.results).as_posix(),
                  "catalog_manifest": (folder / "input.json").relative_to(self.results).as_posix(),
                  "catalog_key": product["key"]}
        config["bank_sha256"] = file_hash(banks / bank_name)
        artifacts = [banks / bank_name, optimized / artifact_name,
                     optimized / artifact_name.replace(".json", ".npz"),
                     folder / (product["key"] + "_split.json"), folder / (product["key"] + "_features.npz")]
        write(folder / "artifact_hashes.json", {p.relative_to(self.results).as_posix(): file_hash(p) for p in artifacts})
        report = {"n": len(samples), "mode": mode, "threshold": low, "t_high": high,
                  "counts": {r: len(indices[r]) for r in ROLES}, "sealed_test_n": sum(s["role"] == "test" for s in product["samples"]),
                  "calibration": measurement(ys, ss, low, high),
                  "qualification": measurement(labels[indices["qualification"]], scores[indices["qualification"]], low, high),
                  "source": product["source"], "scope": "按来源标签的外观验证；未读取封存测试图，实物身份未知时无法保证跨实物独立"}
        from products.qualification import catalog_profile
        qualification = catalog_profile(self.results, config)
        report["automation"] = {k: qualification[k] for k in ("allow_normal", "allow_isolate", "auto_low", "auto_high", "regression", "rule")}
        return report, config

    def activate(self, key, build):
        if not re.fullmatch(r"[a-f0-9]{32}", build):
            raise ValueError("建库版本标识无效")
        with self.lock:
            folder = self.folder(key) / "builds" / build
            state, product = read(folder / "state.json"), read(self.folder(key) / "product.json")
            if state["status"] != "completed" or state["revision"] != product["revision"]:
                raise ValueError("建库未完成或样本已变化，请重新建库")
            config = read(folder / "candidate_config.json")
            self.validate_identities(read(folder / "input.json"))
            registry = read(self.registry) if self.registry.exists() else []
            existing = next((p for p in registry if p["key"] == key), None)
            if existing and digest(existing) != digest(config):
                raise ValueError("该产品已接入其他版本，请使用模型迭代管理后续版本")
            for relative, expected in read(folder / "artifact_hashes.json").items():
                if file_hash(self.results / relative) != expected:
                    raise ValueError("建库文件已变化，禁止发布")
            for sample in read(folder / "input.json")["samples"]:
                if image_hash(self.root / sample["path"]) != sample["pixel_hash"]:
                    raise ValueError("样本已变化，禁止发布旧建库版本")
            if not existing:
                write(self.registry, registry + [config])
            product["active_build"] = build
            write(self.folder(key) / "product.json", product)
            return {"ok": True, "product": config, "message": "产品已接入，自动处置资格由独立验证门槛决定"}
