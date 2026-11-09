"""Durable single-worker batches with frozen recipes and recoverable item attempts."""
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import re
import shutil
import threading
import uuid
from pathlib import Path

from PIL import Image

from review.workflow import digest, image_hash, now, read, write

BATCH_CONTEXT = ContextVar("inspection_batch_context", default={})


class BatchService:
    def __init__(self, results, recipe, detect, recover, evaluate=None):
        self.results = Path(results)
        self.folder = self.results / "batches"
        self.recipe, self.detect, self.recover = recipe, detect, recover
        self.evaluate = evaluate
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspection-batch")
        self.futures = {}
        for path in sorted(self.folder.glob("*/state.json")):
            state = read(path)
            if state["status"] in ("queued", "running", "importing"):
                action = self.import_and_run if state.get("import_rows") else self.execute
                self.futures[path.parent.name] = self.executor.submit(action, path.parent.name)

    def path(self, key):
        if not re.fullmatch(r"[a-f0-9]{32}", key):
            raise ValueError("批次标识无效")
        return self.folder / key

    def create(self, product, purpose="production", name=""):
        if purpose not in ("production", "evaluation", "simulation"):
            raise ValueError("批次用途无效")
        if str(product).startswith('family:') and purpose=='evaluation':
            raise ValueError('产品族批次用于作业或流程演示；质量封存评估必须指定具体检测配置')
        config = self.recipe(product)
        with self.lock:
            pending = sum(read(p)["status"] in ("draft", "queued", "running", "paused", "importing") for p in self.folder.glob("*/state.json"))
            if pending >= 20:
                raise ValueError("请先完成已有批次；最多保留20个未完成批次")
            key = uuid.uuid4().hex
            write(self.path(key) / "state.json", {"id": key, "product": product, "name": str(name).strip()[:100] or "检验批次",
                "purpose": purpose, "status": "draft", "created": now(), "recipe": config,
                "recipe_digest": digest(config), "total": 0, "message": "等待导入图片"})
        return self.detail(key)

    def add(self, key, content, filename, internal=False):
        if not content or len(content) > 10 * 1024 * 1024:
            raise ValueError("图片须小于10MB")
        with Image.open(io.BytesIO(content)) as source:
            if source.width * source.height > 20_000_000 or source.format not in ("PNG", "JPEG", "WEBP", "BMP"):
                raise ValueError("图片格式或尺寸无效")
            image = source.convert("RGB")
        fingerprint = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        if encoded.tell() > 10 * 1024 * 1024:
            raise ValueError("规范化后的图片大于10MB，请降低分辨率后上传")
        with self.lock:
            state = read(self.path(key) / "state.json")
            if state["status"] != "draft" and not (internal and state["status"] == "importing"):
                raise ValueError("批次已开始，不能继续导入")
            state["total"] = len(self.items(key))
            existing = self.path(key) / "items" / (fingerprint + ".json")
            if existing.exists(): return {"ok": True, "duplicate": True, "id": fingerprint}
            if state["total"] >= 1000: raise ValueError("单批次最多1000张图片")
            storage = sum(p.stat().st_size for p in (self.path(key) / "images").glob("*.png"))
            if storage + encoded.tell() > 2 * 1024 ** 3 or shutil.disk_usage(self.results).free < 2 * 1024 ** 3:
                raise ValueError("批次图片存储已达2GB或剩余磁盘不足2GB")
            image_path = self.path(key) / "images" / (fingerprint + ".png")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(encoded.getbuffer())
            write(existing, {"id": fingerprint, "name": Path(str(filename)).name[:160], "order": state["total"],
                  "status": "pending", "attempt": 0, "image": str(image_path), "result": None})
            state["total"] += 1
            write(self.path(key) / "state.json", state)
        return {"ok": True, "duplicate": False, "id": fingerprint}

    def items(self, key):
        return sorted((read(p) for p in (self.path(key) / "items").glob("*.json")), key=lambda x: x["order"])

    def detail(self, key):
        state = read(self.path(key) / "state.json")
        items = self.items(key)
        totals = {"completed": 0, "failed": 0, "automatic": 0, "review": 0, "vlm": 0}
        result = []
        for item in items:
            value = item["result"] or {}
            totals["completed"] += item["status"] == "completed"
            totals["failed"] += item["status"] == "failed"
            action = value.get("disposition") or {}
            totals["automatic"] += bool(action.get("auto_result")) and not action.get("requires_review", True)
            totals["review"] += bool(action.get("requires_review"))
            totals["vlm"] += bool(value.get("vlm_called"))
            result.append({k: v for k, v in item.items() if k != "image"})
        public_state = {k: v for k, v in state.items() if k not in ("import_rows", "source_root")}
        response = {"ok": True, **public_state, "total": len(items), "counts": totals, "items": result}
        if state["purpose"] == "evaluation" and state["status"] in ("completed", "completed_with_errors"):
            scored = [i for i in items if i["status"] == "completed" and i.get("dataset_truth") in ("normal", "defect")]
            response["evaluation"] = {
                "scored": len(scored), "normal": sum(i["dataset_truth"] == "normal" for i in scored),
                "defect": sum(i["dataset_truth"] == "defect" for i in scored),
                "local_false_positives": sum(i["dataset_truth"] == "normal" and i["result"].get("perception_prediction") == "defect" for i in scored),
                "local_false_negatives": sum(i["dataset_truth"] == "defect" and i["result"].get("perception_prediction") == "normal" for i in scored),
                "local_classified": sum(i["result"].get("perception_prediction") in ("normal", "defect") for i in scored),
                "automatic_normal_misses": sum(i["dataset_truth"] == "defect" and (i["result"].get("disposition") or {}).get("auto_result") == "normal" for i in scored),
                "automatic_isolation_false_positives": sum(i["dataset_truth"] == "normal" and (i["result"].get("disposition") or {}).get("auto_result") == "defect" for i in scored),
                "scope": "与来源标签对照，待复核不计作判对；自动判断错误含抽查前建议"}
        return response

    def list(self):
        return [self.detail(p.parent.name) for p in sorted(self.folder.glob("*/state.json"),
                key=lambda p: p.stat().st_mtime_ns, reverse=True)[:30]]

    def active(self, product=None):
        return any((not product or read(p)["product"] == product) and read(p)["status"] in ("queued", "running", "paused", "importing")
                   for p in self.folder.glob("*/state.json"))

    def start(self, key, retry_failed=False):
        with self.lock:
            state = read(self.path(key) / "state.json")
            if state["status"] == "cancelled": raise ValueError("批次已终止，请创建新批次")
            if key in self.futures and not self.futures[key].done():
                raise ValueError("该批次正在处理，请等待当前操作完成")
            if not self.items(key) and not state.get("import_rows"): raise ValueError("请先导入图片")
            if state["recipe_digest"] != digest(self.recipe(state["product"])):
                raise ValueError("产品配置已变化，请创建新批次，避免混用版本")
            if retry_failed:
                for item in self.items(key):
                    if item["status"] == "failed":
                        item["status"] = "pending"
                        write(self.path(key) / "items" / (item["id"] + ".json"), item)
            state.update(status="queued", message="等待后台检测", updated=now())
            write(self.path(key) / "state.json", state)
            self.futures[key] = self.executor.submit(self.import_and_run if state.get("import_rows") else self.execute, key)
        return {"ok": True, "id": key}

    def pause(self, key):
        with self.lock:
            state = read(self.path(key) / "state.json")
            if state["status"] not in ("queued", "running", "importing"): raise ValueError("该批次未在运行")
            state.update(status="paused", message="已暂停；当前图片完成后停止", updated=now())
            write(self.path(key) / "state.json", state)
        return {"ok": True}

    def from_catalog(self, product, samples, root, source, purpose):
        if not samples: raise ValueError("当前用途尚无样本")
        if len(samples) > 1000: raise ValueError("单次最多1000张，请分批核验")
        signature = digest({"recipe": self.recipe(product), "purpose": purpose, "images": [s["pixel_hash"] for s in samples]})
        with self.lock:
            for path in self.folder.glob("*/state.json"):
                existing = read(path)
                if existing.get("source_signature") == signature and existing["status"] != "cancelled":
                    return {"ok": True, "id": existing["id"], "existing": True}
            job = self.create(product, purpose, "封存测试" if purpose == "evaluation" else "公开标签核验演示")
            key = job["id"]
            state = read(self.path(key) / "state.json")
            state.update(status="importing", source_root=str(root), import_rows=samples, dataset_source=source,
                         message="正在准备已导入的样本", expected_total=len(samples), source_signature=signature)
            write(self.path(key) / "state.json", state)
            self.futures[key] = self.executor.submit(self.import_and_run, key)
        return {"ok": True, "id": key}

    def import_and_run(self, key):
        try:
            state = read(self.path(key) / "state.json")
            for index, sample in enumerate(state["import_rows"]):
                with self.lock:
                    state = read(self.path(key) / "state.json")
                    if state["status"] in ("paused", "cancelled"): return
                    state.update(status="importing", message=f"准备样本 {index+1}/{len(state['import_rows'])}")
                    write(self.path(key) / "state.json", state)
                path = (Path(state["source_root"]) / sample["path"]).resolve()
                if not path.is_relative_to(Path(state["source_root"]).resolve()):
                    raise ValueError("样本路径越界")
                if image_hash(path) != sample["pixel_hash"]: raise ValueError("来源样本已变化")
                added = self.add(key, path.read_bytes(), sample["original_name"], internal=True)
                with self.lock:
                    item_path = self.path(key) / "items" / (added["id"] + ".json")
                    item = read(item_path)
                    item.update(dataset_truth=sample["label"], dataset_source=state["dataset_source"])
                    write(item_path, item)
            self.execute(key)
        except Exception:
            with self.lock:
                state = read(self.path(key) / "state.json")
                state.update(status="failed", message="样本准备失败，请检查来源文件后继续")
                write(self.path(key) / "state.json", state)

    def cancel(self, key):
        with self.lock:
            state = read(self.path(key) / "state.json")
            if state["status"] in ("completed", "completed_with_errors"):
                raise ValueError("批次已完成")
            state.update(status="cancelled", message="批次已终止，既有结果保留", updated=now())
            write(self.path(key) / "state.json", state)
        return {"ok": True}

    def execute(self, key):
        try:
            self.run_items(key)
        except Exception:
            with self.lock:
                state = read(self.path(key) / "state.json")
                state.update(status="failed", message="批次准备失败，请检查配套产品版本后重试", updated=now())
                write(self.path(key) / "state.json", state)

    def run_items(self, key):
        items = self.items(key)
        for item in items:
            if item["status"] in ("completed", "failed"): continue
            with self.lock:
                state = read(self.path(key) / "state.json")
                if state["status"] in ("paused", "cancelled"): return
                try:current_recipe=digest(self.recipe(state['product']))
                except Exception:
                    state.update(status='paused',message='当前匹配或检测配置暂不可用，已暂停并保留已完成结果')
                    write(self.path(key)/'state.json',state)
                    return
                if state["recipe_digest"] != current_recipe:
                    state.update(status="paused", message="产品版本变化，已暂停以避免混用版本")
                    write(self.path(key) / "state.json", state)
                    return
                was_running = item["status"] == "running"
                if not was_running: item["attempt"] += 1
                item["status"] = "running"
                write(self.path(key) / "items" / (item["id"] + ".json"), item)
                state.update(status="running", message=f"正在检测 {item['order'] + 1}/{len(items)}", updated=now())
                write(self.path(key) / "state.json", state)
            context = {"batch_id": key, "batch_item_id": item["id"], "batch_attempt": item["attempt"],
                       "record_scope": state["purpose"]}
            if state["purpose"] == "simulation":
                context["source_annotation"] = {"source": item.get("dataset_source", ""), "label": item.get("dataset_truth"),
                                                 "method": "public_dataset_reference"}
            token = BATCH_CONTEXT.set(context)
            try:
                result = self.recover(context) if was_running else None
                if result is None:
                    if image_hash(item["image"]) != item["id"]:
                        raise ValueError("批次原图已变化")
                    result = self.detect(Path(item["image"]).read_bytes(), item["name"], state["product"])
                # Original image remains retrievable separately; don't duplicate large image payloads in state.
                item["result"] = {k: v for k, v in result.items() if k not in ("heatmap", "classification")}
                item["status"] = "completed" if result.get("ok") else "failed"
            except Exception:
                item["status"] = "failed"
                item["result"] = {"ok": False, "error": "检测失败，可重试失败项目"}
            finally:
                BATCH_CONTEXT.reset(token)
            with self.lock:
                write(self.path(key) / "items" / (item["id"] + ".json"), item)
        with self.lock:
            state = read(self.path(key) / "state.json")
            if state["status"] == "cancelled": return
            failures = sum(item["status"] == "failed" for item in self.items(key))
            state.update(status="completed_with_errors" if failures else "completed", finished=now(),
                         message="检测完成，部分项目可重试" if failures else "批次检测完成")
            write(self.path(key) / "state.json", state)
            if state["purpose"] == "evaluation" and self.evaluate:
                try:
                    state["release_evaluation"] = self.evaluate(self.detail(key))
                except Exception as exc:
                    state["release_evaluation"] = {"ok": False, "error": str(exc), "automatic_release_vetoed": True}
                write(self.path(key) / "state.json", state)
