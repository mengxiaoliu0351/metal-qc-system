# -*- coding: utf-8 -*-
"""FastAPI后端：包装检测逻辑，提供REST接口给HTML前端"""
import os, sys, io, json, base64, time, threading
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from PIL import Image
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import THRESHOLD, RESULTS_DIR
from gating.router import gate
from cognition.rag import KnowledgeBase
from cognition import vlm_api
from web.public_demo import PUBLIC_REQUEST, PublicDemoMiddleware

app = FastAPI(title="智能质检系统API")
from web.family_api import router as family_router
app.include_router(family_router)
from web.body_quality_api import router as body_quality_router
app.include_router(body_quality_router)
from web.family_inspection import router as family_inspection_router
app.include_router(family_inspection_router)
from web.screw_delivery_api import router as screw_delivery_router
app.include_router(screw_delivery_router)
if os.environ.get("PUBLIC_DEMO", "0") == "1":
    app.add_middleware(PublicDemoMiddleware, admin_token=os.environ.get("INSPECTION_ADMIN_TOKEN", ""),
                       results_dir=RESULTS_DIR, daily_limit=int(os.environ.get("PUBLIC_DAILY_LIMIT", "60")))
else:
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
if os.environ.get("DEMO_LOGIN", "0") == "1":
    from web.demo_login import DemoLoginMiddleware
    app.add_middleware(DemoLoginMiddleware, admin_token=os.environ.get("INSPECTION_ADMIN_TOKEN", ""),
                       username=os.environ.get("DEMO_USERNAME", "demo"), salt=os.environ.get("DEMO_PASSWORD_SALT", ""),
                       digest=os.environ.get("DEMO_PASSWORD_HASH", ""), require_https=True,
                       password_required=os.environ.get("DEMO_PASSWORD_REQUIRED", "1") != "0")
_inference_lock = threading.RLock()


@app.get("/healthz")
async def health():
    return {"ok": True, "service": "metalqc"}

# 全局加载模型（启动时加载一次）
_engine = None
_kb = None
_sem_kb = None
_clfs = {}
_products = None
_current_product = None
_optimized_models = {}
_patchcore_features = None
_review_workflow = None
_product_catalog = None
_batch_service = None
_recipe_hashes = {}

BANKS_DIR = os.path.join(RESULTS_DIR, "perception", "banks")


def get_review_workflow():
    global _review_workflow
    from review.workflow import ReviewWorkflow
    if _review_workflow is None or str(_review_workflow.results) != str(RESULTS_DIR):
        def extract_review_features(product, paths):
            from perception.feature_backends import cosine_anomalies
            # Reuse CLIP; locking prevents concurrent bank changes and hook collisions.
            with _inference_lock:
                detector, config = get_engine(product)
                bank = np.load(os.path.join(RESULTS_DIR, "perception", "banks", config["bank"]))
                features, scores = [], []
                for start in range(0, len(paths), 2):
                    images = []
                    for path in paths[start:start + 2]:
                        with Image.open(path) as image:
                            images.append(image.convert("RGB"))
                    patches, vectors = detector.extract_batch(images)
                    features.extend(vectors)
                    scores.extend(float(cosine_anomalies(x, bank).max()) for x in patches)
                return np.asarray(features), np.asarray(scores)
        _review_workflow = ReviewWorkflow(ROOT, RESULTS_DIR, extractor=extract_review_features)
    return _review_workflow


def workflow_call(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def get_product_catalog():
    global _product_catalog
    from products.catalog import ProductCatalog
    if _product_catalog is None or str(_product_catalog.results) != str(RESULTS_DIR):
        def extract(samples, progress):
            from perception.feature_backends import cosine_anomalies
            vectors = [None] * len(samples)
            scores = np.zeros(len(samples))
            bank_indices = [i for i, s in enumerate(samples) if s["role"] == "bank"]
            others = [i for i, s in enumerate(samples) if s["role"] != "bank"]
            patches = []
            done = 0
            for indices in (bank_indices, others):
                for start in range(0, len(indices), 2):
                    current = indices[start:start + 2]
                    with _inference_lock:
                        detector, _ = get_engine(load_products()[0]["key"])
                        images = []
                        for i in current:
                            with Image.open(os.path.join(ROOT, samples[i]["path"])) as image:
                                images.append(image.convert("RGB"))
                        local, global_vectors = detector.extract_batch(images)
                    for i, local_features, vector in zip(current, local, global_vectors):
                        vectors[i] = vector
                        if samples[i]["role"] == "bank":
                            patches.append(local_features)
                        else:
                            scores[i] = float(cosine_anomalies(local_features, bank).max())
                    done += len(current)
                    progress(done, len(samples))
                if indices is bank_indices:
                    bank = np.concatenate(patches, axis=0)
            return {"bank": bank, "global_features": np.asarray(vectors), "clip_scores": scores}
        _product_catalog = ProductCatalog(ROOT, RESULTS_DIR, extractor=extract, lock=get_review_workflow().lock)
    return _product_catalog


def get_batch_service():
    global _batch_service
    from review.batches import BatchService
    from review.workflow import file_hash
    if _batch_service is None or str(_batch_service.results) != str(RESULTS_DIR):
        def recipe(product):
            if isinstance(product,str) and product.startswith('family:'):
                from web.family_inspection import batch_recipe
                return batch_recipe(product.split(':',1)[1],recipe)
            config = next((p for p in load_products() if p["key"] == product), None)
            if config is None: raise ValueError("请选择已接入的产品")
            paths = [os.path.join(BANKS_DIR, config["bank"])]
            if config.get("perception_artifact"):
                meta = os.path.join(RESULTS_DIR, "perception/optimized", config["perception_artifact"])
                with open(meta, encoding="utf8") as stream:
                    model_metadata = json.load(stream)
                paths += [meta, os.path.join(os.path.dirname(meta), model_metadata["arrays"])]
            for relative in (f"results/perception/classifier_{product}.joblib", "src/cognition/vlm_api.py",
                             "src/review/disposition.py", "src/perception/winclip_patch.py"):
                path = os.path.join(ROOT, relative)
                if os.path.isfile(path): paths.append(path)
            hashes = {}
            for path in paths:
                stat = os.stat(path)
                stamp = (path, stat.st_size, stat.st_mtime_ns)
                if stamp not in _recipe_hashes: _recipe_hashes[stamp] = file_hash(path)
                hashes[os.path.relpath(path, RESULTS_DIR)] = _recipe_hashes[stamp]
            return {"config": config, "files": hashes}
        def detect(content, filename, product):
            if product.startswith('family:'):
                from web.family_inspection import inspect
                return inspect(content,filename,product.split(':',1)[1])
            return json.loads(_locked_detection(content, filename, product).body)
        def recover(context):
            for _, record in get_review_workflow().records():
                if all(record.get(k) == v for k, v in context.items()):
                    gate = record["gate"]
                    return {**{k: record.get(k) for k in ("session_id", "score", "cognition", "disposition", "time_total", "vlm_called", "perception_prediction", "baseline_score", "perception_method", "quality_check", "family_key", "family_name", "family_routing", "family_action", "quality_model_bound", "iteration_eligible")},
                            "ok": gate["route"] != "processing_error", "route": gate["route"],
                            "threshold": gate.get("threshold"), "t_high": gate.get("t_high")}
            return None
        _batch_service = BatchService(RESULTS_DIR, recipe, detect, recover,
                                      evaluate=get_review_workflow().policy().observe_evaluation)
    return _batch_service


@app.on_event("startup")
def restore_batch_jobs():
    get_batch_service()


def reset_product_cache():
    global _products
    _products = None
    _optimized_models.clear()


def apply_optimized_perception(product_key, config, engine, image, result):
    """Only enable artifacts explicitly selected in products.json after evaluation."""
    artifact = config.get("perception_artifact")
    if not artifact:
        return {**result, "method": "CLIP patch 原始基线"}
    if product_key not in _optimized_models:
        from perception.feature_backends import OptimizedPerception, PatchCoreFeatures

        def shared_patchcore():
            global _patchcore_features
            if _patchcore_features is None:
                _patchcore_features = PatchCoreFeatures()
            return _patchcore_features

        model = OptimizedPerception(os.path.join(RESULTS_DIR, "perception", "optimized", artifact),
                                    patchcore_factory=shared_patchcore)
        if model.metadata["product"] != product_key:
            raise ValueError("优化模型与产品不匹配")
        if not np.isclose(float(model.metadata["threshold"]), float(config["threshold"]), rtol=0, atol=1e-10):
            raise ValueError("优化模型与检测阈值不匹配，请恢复成套配置")
        _optimized_models[product_key] = model
    return _optimized_models[product_key].apply(engine, image, result)


def load_products():
    """加载产品配置 products.json"""
    global _products
    if _products is None:
        cfg_path = os.path.join(BANKS_DIR, "products.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                _products = json.load(f)
        else:
            # 兜底：扫描banks目录
            _products = [{
                "key": "metal_nut",
                "name": "金属螺母 (MVTec metal_nut)",
                "bank": "金属螺母(metal_nut).npy",
                "threshold": THRESHOLD,
            }]
    return _products


def get_engine(product_key: str = "metal_nut"):
    """加载检测引擎，并切换到指定产品的记忆库与阈值。
    返回 (engine, product_config)。"""
    global _engine, _current_product
    products = load_products()
    pcfg = next((p for p in products if p["key"] == product_key), None)
    if pcfg is None:
        raise ValueError("未知产品，请选择与图片一致的已建库产品")
    if _engine is None:
        from perception.winclip_patch import WinCLIPPatch
        _engine = WinCLIPPatch("ViT-B-16")

    if _current_product != product_key:
        bank_path = os.path.join(BANKS_DIR, pcfg["bank"])
        _engine.bank = np.load(bank_path)
        _engine.bank_info = {"n_images": _engine.bank.shape[0] // 196,
                             "n_feats": _engine.bank.shape[0]}
        _current_product = product_key
    return _engine, pcfg


def get_kb():
    global _kb
    if _kb is None:
        _kb = KnowledgeBase(csv_path=os.path.join(RESULTS_DIR, "cognition", "standard_library.csv"))
    return _kb


def get_clf(product_key: str = "metal_nut"):
    """按产品加载监督缺陷类型分类器（#1）。各产品独立训练，缺失则返回None"""
    global _clfs
    if product_key not in _clfs:
        from perception.defect_classifier import DefectClassifier
        p = os.path.join(RESULTS_DIR, "perception", f"classifier_{product_key}.joblib")
        _clfs[product_key] = DefectClassifier(p)
    return _clfs[product_key]


def get_sem_kb(engine):
    """Chinese notes use character retrieval rather than English CLIP text embeddings."""
    global _sem_kb
    if _sem_kb is None:
        _sem_kb = KnowledgeBase()
    return _sem_kb

def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def product_high_threshold(pcfg):
    """Expose the same effective high threshold to the UI and detection route."""
    low = float(pcfg["threshold"])
    cal = pcfg.get("calibration", {}) or {}
    median = float(cal.get("median", 0) or 0)
    mad = float(cal.get("mad", 0) or 0)
    high = float(pcfg.get("t_high", 0) or 0)
    if high <= 0 and median > 0 and mad > 0:
        high = round(median + 3.0 * mad, 4)
    if high <= low:
        high = round(low * 1.5, 4)
    return high

@app.get("/api/products")
async def list_products():
    """列出可用产品记忆库 + VLM通用兜底模式"""
    products = load_products()
    from web.family_inspection import inspection_targets
    return JSONResponse({
        "inspection_targets":inspection_targets(products,RESULTS_DIR),
        "products": [{"key": p["key"], "name": p["name"],
                      "threshold": p["threshold"],
                      "t_high": product_high_threshold(p),
                      "method": p.get("perception_name", "CLIP patch 原始基线")} for p in products],
        "fallback": {"key": "vlm_general",
                     "name": "通用直检（未建库产品 · AI视觉兜底）"},
    })


@app.get("/api/product-catalog")
def product_catalog_list():
    products = get_product_catalog().list()
    policy = get_review_workflow().policy()
    configs = {c["key"]: c for c in load_products()}
    holds = policy.settings().get("holds", {})
    for product in products:
        if product["key"] not in configs: continue
        try:
            profile = policy.profile(configs[product["key"]])
            hold = holds.get(product["key"] + ":" + profile["model_key"])
            product["runtime_automation"] = {"held": bool(hold), "reason": (hold or {}).get("reason", ""),
                "allow_normal": bool(profile["allow_normal"] and not hold),
                "allow_isolate": bool(profile["allow_isolate"] and not hold)}
        except (OSError, ValueError, KeyError, TypeError):
            product["runtime_automation"] = {"held": True, "reason": "当前版本缺少配套评估", "allow_normal": False, "allow_isolate": False}
    return {"ok": True, "products": products}


@app.post("/api/product-catalog")
def product_catalog_create(request: dict):
    return workflow_call(get_product_catalog().create, request)


@app.get("/api/product-catalog/{key}")
def product_catalog_detail(key: str):
    return workflow_call(get_product_catalog().detail, key)


@app.post("/api/product-catalog/{key}/batches")
def catalog_sample_batch(key: str, request: dict):
    from review.workflow import read
    catalog = get_product_catalog()
    product = workflow_call(read, catalog.folder(key) / "product.json")
    if not product.get("active_build"): raise HTTPException(400, "请先完成建库并接入产品")
    role = request.get("role", "test")
    if role not in ("test", "feedback"): raise HTTPException(400, "请选择封存测试或后续核验样本")
    samples = [s for s in product["samples"] if s["role"] == role]
    if role == "feedback":
        normal = sorted((s for s in samples if s["label"] == "normal"), key=lambda x: x["pixel_hash"])[:20]
        defects = sorted((s for s in samples if s["label"] == "defect"), key=lambda x: x["pixel_hash"])[:20]
        samples = normal + defects
    return workflow_call(get_batch_service().from_catalog, key, samples, ROOT, product["source"],
                         "evaluation" if role == "test" else "simulation")


@app.post("/api/product-catalog/{key}/samples")
async def product_catalog_samples(key: str, file: UploadFile = File(...), role: str = Form(...),
                                  label: str = Form(...), group_id: str = Form("")):
    content = await file.read(10 * 1024 * 1024 + 1)
    return await run_in_threadpool(workflow_call, get_product_catalog().add_image,
                                  key, content, file.filename, role, label, group_id)


@app.post("/api/product-catalog/{key}/builds")
def product_catalog_build(key: str, request: dict):
    return workflow_call(get_product_catalog().start, key, request.get("mode", "normal_memory"))


@app.post("/api/product-catalog/{key}/builds/{build}/activate")
def product_catalog_activate(key: str, build: str):
    with _inference_lock:
        result = workflow_call(get_product_catalog().activate, key, build)
        reset_product_cache()
        return result


@app.get("/api/batches")
def batch_list():
    return {"ok": True, "batches": [{k: v for k, v in b.items() if k != "items"} for b in get_batch_service().list()]}


@app.post("/api/batches")
def batch_create(request: dict):
    return workflow_call(get_batch_service().create, request.get("product", ""), request.get("purpose", "production"), request.get("name", ""))


@app.get("/api/batches/{key}")
def batch_detail(key: str):
    return workflow_call(get_batch_service().detail, key)


@app.post("/api/batches/{key}/items")
async def batch_add(key: str, file: UploadFile = File(...)):
    content = await file.read(10 * 1024 * 1024 + 1)
    return await run_in_threadpool(workflow_call, get_batch_service().add, key, content, file.filename)


@app.get("/api/batches/{key}/items/{item}/image")
def batch_image(key: str, item: str):
    import re
    if not re.fullmatch(r"[a-f0-9]{64}", item): raise HTTPException(400, "图片标识无效")
    folder = workflow_call(get_batch_service().path, key)
    path = os.path.join(folder, "images", item + ".png")
    if not os.path.isfile(path): raise HTTPException(404, "未找到批次图片")
    return FileResponse(path, media_type="image/png")


@app.post("/api/batches/{key}/start")
def batch_start(key: str, request: dict = None):
    return workflow_call(get_batch_service().start, key, bool((request or {}).get("retry_failed")))


@app.post("/api/batches/{key}/pause")
def batch_pause(key: str):
    return workflow_call(get_batch_service().pause, key)


@app.post("/api/batches/{key}/cancel")
def batch_cancel(key: str):
    return workflow_call(get_batch_service().cancel, key)


@app.get("/api/inspection/policy")
async def inspection_policy():
    return get_review_workflow().policy().public_settings()


@app.post("/api/inspection/policy")
async def configure_inspection_policy(request: dict):
    if get_batch_service().active(): raise HTTPException(409, "请等待后台批次结束后调整作业策略")
    return workflow_call(get_review_workflow().policy().configure, request)


@app.get("/api/inspection/evidence")
def inspection_evidence():
    result = []
    for config in load_products():
        try:
            profile = get_review_workflow().policy().profile(config)
            result.append({k: v for k, v in profile.items() if k != "centroid"})
        except (OSError, ValueError, KeyError) as exc:
            result.append({"product": config["key"], "allow_normal": False, "allow_isolate": False,
                           "reason": "当前版本缺少配套的自动处置评估"})
    return {"ok": True, "profiles": result}


@app.get("/api/inspection/gray-evidence")
def gray_inspection_evidence():
    from review.metrics import gray_decision_summary
    return {"ok": True, **gray_decision_summary(rec for _, rec in get_review_workflow().records())}


@app.get("/api/inspection/audit")
def inspection_audit():
    from review.metrics import audit_window_summary
    workflow = get_review_workflow()
    period = workflow.policy().settings().get("audit_period", "day")
    key = workflow.policy().period_key(period)
    return {"ok": True, "audit_period": period, "sample_rate": workflow.policy().settings()["sample_rate"],
            **audit_window_summary((rec for _, rec in workflow.records()), key)}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), product: str = Form("metal_nut")):
    """检测接口：上传图片，返回异常分数、热力图、缺陷类型、VLM分析。
    product=vlm_general 时走VLM通用直检兜底（不依赖记忆库）。"""
    content = await file.read(10 * 1024 * 1024 + 1)
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "图片不能超过10MB")
    return await run_in_threadpool(_locked_detection, content, file.filename, product)


def _locked_detection(content, filename, product):
    with _inference_lock:
        return _detect_image(content, filename, product)


def _finalize_detection(img, filename, config, inference, started, payload):
    """Persist machine action independently of human truth, for every valid input path."""
    from review.workflow import image_hash, write
    from review.disposition import image_quality
    from interface import report
    import uuid
    session = datetime.now().strftime("%H%M%S_") + uuid.uuid4().hex[:10]
    day = datetime.now().strftime("%Y-%m-%d")
    directory = os.path.join(RESULTS_DIR, "history", "images", day)
    os.makedirs(directory, exist_ok=True)
    input_path = os.path.join(directory, session + "_input.png")
    img.save(input_path)
    heatmap_path = None
    if inference.get("heatmap") is not None:
        heatmap_path = os.path.join(directory, session + "_heatmap.png")
        inference["heatmap"].save(heatmap_path)
    record = {"session_id": session, "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "score": payload.get("score"),
              "gate": {"route": payload["route"], "label": payload.get("route_label"),
                       "threshold": payload.get("threshold"), "t_high": payload.get("t_high")},
              "cognition": payload.get("cognition") or {}, "references": payload.get("references") or [],
              "classification": payload.get("classification"), "supervised_class": payload.get("supervised_class"),
              "perception_method": payload.get("perception_method"), "baseline_score": payload.get("baseline_score"),
              "perception_prediction": payload.get("perception_prediction"),
              "perception_version": (config or {}).get("review_version", "initial"),
              "input_png": input_path, "heatmap_png": heatmap_path,
              "product_key": (config or {}).get("key", "vlm_general"),
              "product_name": (config or {}).get("name", "通用视觉直检 / 未建库产品"),
              "image_hash": image_hash(input_path), "quality_check": image_quality(img),
              "vlm_called": bool(payload.get("vlm_called"))}
    from review.batches import BATCH_CONTEXT
    record.update(BATCH_CONTEXT.get())
    from review.inspection_context import INSPECTION_CONTEXT
    family_context=dict(INSPECTION_CONTEXT.get());started=family_context.pop('_started',started)
    record.update(family_context)
    record["disposition"] = get_review_workflow().policy().decide(record, config, inference.get("image_features"))
    if record.get("record_scope") == "simulation":
        record["disposition"].update(status="needs_review", requires_review=True, label="公开标签核验演示 · 待核验",
                                     reason="请对照来源标签完成核验；该记录不计入日常作业质量", reason_code="simulation_review")
        if record.get('family_key'):
            record['disposition'].update(label='产品族流程演示 · 待核验',reason='固定样例用于核对自动匹配和完整流程，不提供人工质量真值，也不计入日常质效')
    record["time_total"] = round(time.perf_counter() - started, 3)
    report.save_history(record, sample_name=filename or "工件")
    # Legacy tmp metadata remains available for old report consumers.
    tmp_dir = os.path.join(RESULTS_DIR, "perception", "web_tmp")
    write(os.path.join(tmp_dir, session + "_meta.json"), record)
    return JSONResponse({**payload, "session_id": session, "disposition": record["disposition"],
                         "quality_check": record["quality_check"], "time_total": record["time_total"]})


def _detect_image(content, filename, product):
    request_started = time.perf_counter()
    persisting = False
    try:
        # Decode before conversion to reject oversized images without allocating their pixels.
        with Image.open(io.BytesIO(content)) as source:
            if source.width * source.height > 20_000_000:
                raise ValueError("图片像素过大，请压缩至2000万像素以内")
            if source.format not in ("PNG", "JPEG", "WEBP", "BMP"):
                raise ValueError("请上传PNG、JPEG、WEBP或BMP图片")
        os.makedirs(os.path.join(RESULTS_DIR, "perception"), exist_ok=True)
        img = Image.open(io.BytesIO(content)).convert("RGB")
        from review.disposition import image_quality, validate_references
        quality = image_quality(img)
        if not quality["ok"]:
            config = next((p for p in load_products() if p["key"] == product), None)
            persisting = True
            return _finalize_detection(img, filename, config, {}, request_started, {
                "ok": True, "score": None, "threshold": None, "t_high": None,
                "route": "input_review", "route_label": "输入待核验", "heatmap": None,
                "product": (config or {}).get("name", "未建库产品"), "vlm_called": False,
                "cognition": {"ok": False, "conclusion": "需重新采集图像", "defect_desc": "；".join(quality["reasons"]),
                              "suggestion": "请更换清晰、曝光正常的工件图像；本次不执行模型与VLM推理"}})

        # ===== VLM 通用直检兜底（未建库新产品）=====
        if product == "vlm_general":
            t0 = time.time()
            tmp_path = os.path.join(RESULTS_DIR, "perception", "_api_tmp.png")
            img.save(tmp_path)
            prompt = ("该产品尚未建立正常样本库，请直接基于图像本身判断金属表面是否存在可见缺陷"
                      "（划痕、裂纹、凹坑、锈斑、变形、色差等）。按系统规定的JSON格式输出结论，"
                      "ref_ids 若无对应标准可留空数组，suggestion 给出处置建议。")
            cog = vlm_api.analyze(tmp_path, prompt)
            persisting = True
            return _finalize_detection(img, filename, None, {}, request_started, {
                "ok": True, "score": None, "threshold": None, "t_high": None,
                "route": "general", "route_label": "通用直检 · AI视觉分析",
                "heatmap": None, "classification": None, "cognition": cog,
                "vlm_called": True, "time_perception": round(time.time() - t0, 2)})

        # ===== 常规双链路：按产品加载对应记忆库与阈值 =====
        engine, pcfg = get_engine(product)
        prod_threshold = float(pcfg["threshold"])
        t0 = time.time()
        result = engine.detect(img)
        result = apply_optimized_perception(product, pcfg, engine, img, result)
        t_perception = time.time() - t0

        score = float(result["score"])
        perception_prediction = "defect" if score >= prod_threshold else "normal"
        heatmap_b64 = img_to_b64(result["heatmap"])
        classification = result.get("classification", {})

        # #1 监督缺陷类型分类（各产品独立训练，有模型才跑）
        sup_clf = None
        try:
            clf = get_clf(product)
            if clf.available:
                sup_clf = clf.predict(engine, img, image_features=result.get("image_features"))
        except Exception as _e:
            print("监督分类失败:", _e)

        # 2. 三区间门控（低风险/灰区VLM/高风险）
        t_high = product_high_threshold(pcfg)
        g = gate(score, t_low=prod_threshold, t_high=t_high)

        # 3. 认知层：仅灰区调 VLM + 国标 RAG；低/高风险本地直出，省 API 调用
        cognition = None
        retrieved = []
        vlm_called = False
        scope_precheck = None
        if result.get("image_features") is not None:
            try:
                policy = get_review_workflow().policy()
                scope_precheck = policy.scope_check(pcfg, result["image_features"], policy.profile(pcfg))
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if scope_precheck is not None and not scope_precheck["ok"]:
            cognition = {"ok": False, "conclusion": "产品适用性待核验", "defect_desc": scope_precheck["reason"],
                         "suggestion": "请核对工件产品及图像采集条件，本次不调用VLM"}
        elif g["route"] == "vlm_review":
            # 灰区：调 VLM 深度复核 + RAG
            defect_word = "表面缺陷"
            if sup_clf and sup_clf.get("label") and sup_clf["label"] != "正常":
                defect_word = sup_clf["label"]
            elif classification:
                defect_word = max(classification, key=classification.get)
            query = f"{defect_word}金属表面缺陷"
            sem_kb = get_sem_kb(engine)
            retrieved = [dict(r) for r in sem_kb.retrieve(query, top_k=3)]
            from cognition.knowledge_sources import STANDARD_CATALOG
            if product in STANDARD_CATALOG: retrieved.append(dict(STANDARD_CATALOG[product]))
            texts = [f"[{r['id']}] {r['standard']} {r['section']}: {r['text']}" for r in retrieved]
            visual_review = os.environ.get("VLM_FOCUS_REVIEW", "0") == "1"
            hint = (f"当前产品：{pcfg['name']}。请独立观察整图及局部；本地模型将本图分入待分析区。"
                    if visual_review else f"异常分数{score:.4f}（介于{prod_threshold}与{t_high}之间），疑似{defect_word}缺陷")
            prompt = vlm_api.build_prompt(hint, texts)
            tmp_path = os.path.join(RESULTS_DIR, "perception", "_api_tmp.png")
            img.save(tmp_path)
            vlm_called = True
            if visual_review:
                from cognition.visual_input import focus_views
                focus, focus_metadata = focus_views(img, result.get("anomaly_map"), engine.preprocess)
                cognition = validate_references(vlm_api.analyze(
                    tmp_path, prompt, focus_images=focus, visual_review=True), retrieved)
                cognition["visual_input"] = focus_metadata
            else:
                cognition = validate_references(vlm_api.analyze(tmp_path, prompt), retrieved)
            if cognition.get("conclusion") in ("合格", "正常"):
                # Keep the VLM's original opinion; the separate disposition rule may abstain.
                cognition["vlm_conclusion"] = cognition["conclusion"]
                cognition["suggestion"] = "本地分数仍处灰区；须通过作业规则才能自动归档，否则转人工核验"
        elif g["route"] == "high_risk":
            # 高风险：本地直出"明显异常，建议人工复核"，不调 VLM
            top_defect = "未知"
            if sup_clf and sup_clf.get("label") and sup_clf["label"] != "正常":
                top_defect = sup_clf["label"]
            elif classification:
                top_defect = max(classification, key=classification.get)
            confidence = f"（分类置信度{sup_clf['confidence']*100:.0f}%）" if sup_clf else ""
            cognition = {
                "ok": True,
                "conclusion": "疑似不合格",
                "defect_desc": f"检测到明显异常区域，疑似{top_defect}缺陷{confidence}" if top_defect != "正常" else "检测到明显局部异常，建议进一步分析",
                "suggestion": "系统依据作业策略隔离疑似缺陷；抽查及边界样本进入人工核验，不自动执行返工或报废",
            }
        else:
            # 低风险：本地放行，不调 VLM
            cognition = {
                "ok": True,
                "conclusion": "系统判正常",
                "defect_desc": "异常分数低于低风险阈值，未发现明显异常",
                "suggestion": "满足自动归档条件则保存系统判断；临界、抽查及适用性异常样本进入人工核验",
            }

        persisting = True
        return _finalize_detection(img, filename, pcfg, result, request_started, {
            "ok": True, "product": pcfg["name"], "score": score,
            "threshold": prod_threshold, "t_high": t_high, "route": g["route"], "route_label": g["label"],
            "heatmap": heatmap_b64, "classification": classification, "supervised_class": sup_clf,
            "perception_method": result.get("method"), "baseline_score": result.get("baseline_score", score),
            "perception_prediction": perception_prediction, "cognition": cognition, "references": retrieved,
            "vlm_called": vlm_called, "time_perception": round(t_perception, 2)})
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("Detection failed")
        # A decoded workpiece with a failed model still needs a traceable exception task.
        if 'img' in locals() and not persisting:
            try:
                config = next((p for p in load_products() if p['key'] == product), None)
                return _finalize_detection(img, filename, config, {}, request_started, {
                    'ok': True, 'score': None, 'threshold': None, 't_high': None,
                    'route': 'processing_error', 'route_label': '处理异常 · 待复核', 'heatmap': None,
                    'vlm_called': False, 'processing_error': True,
                    'cognition': {'ok': False, 'conclusion': '检测未完成',
                                  'defect_desc': '模型处理失败，本次不作正常或缺陷判断',
                                  'suggestion': '核实产品与服务状态后重试，或人工核验原图'}})
            except Exception:
                logging.getLogger(__name__).exception('Could not persist detection exception')
        return JSONResponse({"ok": False, "error": "检测未完成，请检查图片和产品类型或稍后重试。" if PUBLIC_REQUEST.get() else str(e)})

@app.get("/api/history")
async def history(limit: int = 50):
    """返回历史记录"""
    from interface import report
    from review.disposition import effective_disposition
    records = [{**r, "current_disposition": effective_disposition(r)} for r in report.list_history(limit=limit)]
    return JSONResponse({"records": records})


@app.get("/api/history/detail")
async def history_detail(day: str = "", key: str = ""):
    """回看某条历史详情：返回原图、热力图(base64)、结论等完整字段"""
    from interface import report
    from review.disposition import effective_disposition
    _, rec = workflow_call(get_review_workflow().locate, day=day, key=key)

    def _b64(path):
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return None

    return JSONResponse({
        "record": {**rec, "current_disposition": effective_disposition(rec)},
        "input_img": _b64(rec.get("input_png")),
        "heatmap_img": _b64(rec.get("heatmap_png")),
    })

@app.get("/api/stats")
async def stats():
    """返回统计数据"""
    from interface import report
    records = [r for _, r in get_review_workflow().records() if r.get("record_scope", "production") == "production"][:500]
    from review.metrics import operational_summary
    from review.workflow import original_prediction
    operations = operational_summary(records)
    total = len(records)
    today = datetime.now().strftime("%Y-%m-%d")
    today_recs = [r for r in records if r.get("timestamp","").startswith(today)]
    fail = sum(original_prediction(r) == "defect" for r in records)
    normal_count = sum(original_prediction(r) == "normal" for r in records)
    low_risk = sum(1 for r in records if r.get("gate",{}).get("route") == "low_risk")
    vlm_review = sum(1 for r in records if r.get("gate",{}).get("route") == "vlm_review")
    high_risk = sum(1 for r in records if r.get("gate",{}).get("route") == "high_risk")
    products = load_products()
    real_threshold = float(products[0]["threshold"]) if products else THRESHOLD
    return JSONResponse({
        "total": total,
        "today": len(today_recs),
        "fail": fail,
        "pass": normal_count,
        "unclassified": total - fail - normal_count,
        "operations": operations,
        "low_risk": low_risk,
        "vlm_review": vlm_review,
        "high_risk": high_risk,
        "threshold": real_threshold,
        "product_count": len(products),
    })

@app.post("/api/chat")
def chat(req_data: dict):
    """AI助手问答：多轮对话 + 全量历史聚合统计（按产品/按日期），可答任意范围"""
    import requests as http_req
    from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DASHSCOPE_CHAT_MODEL
    from interface import report as rep

    # 兼容两种入参：多轮 messages 或单轮 question
    messages_in = req_data.get("messages")
    if messages_in:
        question = messages_in[-1].get("content", "")
        history_msgs = messages_in[:-1]
    else:
        question = req_data.get("question", "")
        history_msgs = []

    records = [] if PUBLIC_REQUEST.get() else rep.list_history(limit=500)
    total = len(records)
    today = datetime.now().strftime("%Y-%m-%d")
    today_recs = [r for r in records if r.get("timestamp", "").startswith(today)]
    from review.workflow import original_prediction
    from review.metrics import operational_summary
    fail = sum(original_prediction(r) == 'defect' for r in records)
    pass_ = sum(original_prediction(r) == 'normal' for r in records)
    operations = operational_summary(records)

    # 各产品阈值表
    products = load_products()
    prod_lines = "\n".join([f"- {p['name']}：判定阈值 {p['threshold']}" for p in products])

    # 按产品统计
    prod_stat = {}
    for r in records:
        pn = r.get("product_name", "未知")
        s = prod_stat.setdefault(pn, {"total": 0, "fail": 0, "normal": 0})
        s["total"] += 1
        s['fail'] += int(original_prediction(r) == 'defect')
        s['normal'] += int(original_prediction(r) == 'normal')
    prod_stat_lines = "\n".join(
        f"- {k}：共{v['total']}次，系统判异常{v['fail']}次，系统判正常{v['normal']}次，其余未分类"
        for k, v in prod_stat.items()) or "（暂无记录）"

    # 按日期统计（最近7天）
    from collections import defaultdict
    day_stat = defaultdict(lambda: {"total": 0, "fail": 0})
    for r in records:
        d = (r.get("timestamp", "") or "")[:10]
        if d:
            day_stat[d]["total"] += 1
            if original_prediction(r) == 'defect':
                day_stat[d]["fail"] += 1
    day_lines = "\n".join(
        f"- {d}：共{v['total']}次，系统判异常{v['fail']}次"
        for d, v in sorted(day_stat.items(), reverse=True)[:7]) or "（暂无记录）"

    # 缺陷类型分布
    defect_counter = {}
    for r in records:
        c = (r.get("cognition") or {}).get("defect_desc", "")
        if c and c not in ("未发现明显异常", "—"):
            key = c[:25]
            defect_counter[key] = defect_counter.get(key, 0) + 1
    defect_lines = "\n".join([f"- {k}：{v}次" for k, v in
                             sorted(defect_counter.items(), key=lambda x: -x[1])[:8]]) or "（暂无缺陷记录）"

    sys_p = f"""你是金属表面智能质检系统的内置数据助手，现在可进行多轮对话。请根据下方【系统真实数据】和对话上下文直接回答，用中文简洁作答。
强制规则：
1. 立即直接回答，不要反问、不要让用户补充；可结合对话上下文理解追问（如"那昨天呢"指上一问提到的日期/产品）；
2. 涉及数字必须严格使用下方数据，不要编造；下方数据没有的才回答"暂无相关记录"；
3. 系统判定不是人工真实标签；未知结果不得计作正常。人工核验一致性来自选择性复核，不能当独立测试准确率。
4. 旧历史中的 GB34528 引用已停用，不可作为金属表面检测依据。当前项目知识卡是教学材料，标准目录仅核验编号与名称，不可编造国标原文、页码或尺寸限值。

【全量统计（最近500条）】
- 历史检测总数：{total}；今日：{len(today_recs)} 条；系统判正常 {pass_} 条，系统判异常 {fail} 条，其余 {total-pass_-fail} 条未分类。
- 作业与人工核验统计：{json.dumps(operations, ensure_ascii=False)}

【各产品判定阈值】（异常分数≥阈值判不合格）
{prod_lines}

【各产品检测统计】
{prod_stat_lines}

【按日期统计（最近7天）】
{day_lines}

【常见缺陷类型分布】
{defect_lines}

【判据说明】快链路：分数对照产品阈值；慢链路：灰区调用视觉大模型与知识检索。配套评估通过且远离边界的低、高风险样本可自动归档或隔离疑似缺陷；灰区仅在结构化结论、分数位置和引用编号同时满足规则时自动分类。抽查、临界、仍不确定的灰区与处理异常送人工核验。自动结果不是训练标签，不执行物理放行、返工或报废。"""

    # 组装多轮 messages
    msgs = [{"role": "system", "content": sys_p}]
    for m in history_msgs[-6:]:   # 保留最近3轮，控token
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})

    # ===== 看图解释：附图则切视觉模型 =====
    img_b64 = req_data.get("image_b64")
    det_ctx = req_data.get("det_ctx", "")
    if img_b64:
        from config import DASHSCOPE_MODEL
        vlm_sys = (
            "你是金属表面缺陷质检视觉助手。用户给你一张待检零件图像（可能含异常热力图，"
            "暖色/高亮区代表相对异常响应，正常图也可能出现暖色，不能仅按颜色认定缺陷）。"
            + (f"\n【系统对这张图的检测结果】{det_ctx}" if det_ctx else "")
            + "\n请结合图片用中文自然语言回答：异常大概在什么位置、可能是什么缺陷类型、"
              "为什么这么判、给什么处置建议。简洁口语，不要输出JSON。"
              "未提供经核实的原文时不得引用国标条款或页码；单张无标尺图像不能推断实际尺寸、深度或限值。")
        vlm_msgs = [
            {"role": "system", "content": vlm_sys},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": question or "这张图为什么被判不合格？可能是什么缺陷？"}
            ]}
        ]
        payload = {"model": DASHSCOPE_MODEL, "messages": vlm_msgs,
                   "temperature": 0.2, "enable_thinking": False}
    else:
        msgs.append({"role": "user", "content": question})
        payload = {"model": DASHSCOPE_CHAT_MODEL, "messages": msgs, "temperature": 0.1}
    payload["max_tokens"] = 1024
    try:
        r = http_req.post(f"{DASHSCOPE_BASE_URL}/chat/completions",
                     headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"},
                     json=payload, timeout=120)
        d = r.json()
        if "choices" not in d:
            err = d.get("error", {})
            msg = err.get("message") or d.get("message") or str(d)[:200]
            hint = "（未配置或失效的大模型 API Key）" if "PASTE" in DASHSCOPE_API_KEY or "Invalid" in msg or "Authentication" in msg else ""
            return JSONResponse({"answer": "助手暂时不可用，请稍后重试或联系演示管理员。" if PUBLIC_REQUEST.get() else f"助手暂时不可用：{msg} {hint}"})
        answer = d["choices"][0]["message"]["content"]
        return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse({"answer": "助手请求未完成，请稍后重试。" if PUBLIC_REQUEST.get() else f"请求失败: {str(e)}"})

@app.get("/api/report/{session_id}")
async def report_pdf(session_id: str):
    """生成PDF质检报告（读取检测时保存的元信息，使用对应产品阈值）"""
    from fastapi.responses import FileResponse
    from interface.report import generate_pdf
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session_id):
        raise HTTPException(400, "报告标识无效")
    try:
        _, record = get_review_workflow().locate(session_id=session_id)
    except FileNotFoundError:
        record = None
    if record is not None:
        return saved_report(record)
    tmp_dir = os.path.join(RESULTS_DIR, "perception", "web_tmp")
    input_path = os.path.join(tmp_dir, f"{session_id}_input.png")
    heatmap_path = os.path.join(tmp_dir, f"{session_id}_heatmap.png")
    meta_path = os.path.join(tmp_dir, f"{session_id}_meta.json")
    if not os.path.exists(input_path):
        raise HTTPException(404, "报告不存在或已过期")

    meta = {"product": "金属零件", "threshold": THRESHOLD, "score": 0.0,
            "route": "fast", "cognition": {}}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    result = {
        "score": meta.get("score", 0.0),
        "gate": {"route": meta.get("route", "fast"),
                 "label": meta.get("route_label", ""),
                 "threshold": meta.get("threshold", THRESHOLD), "t_high": meta.get("t_high")},
        "input_png": input_path,
        "heatmap_png": heatmap_path if os.path.exists(heatmap_path) else None,
        "cognition": meta.get("cognition") or {},
        "product_name": meta.get("product", "未记录"),
        "references": meta.get("references") or [],
    }
    pdf_path = os.path.join(tmp_dir, f"{session_id}_report.pdf")
    generate_pdf(result, pdf_path, sample_name=meta.get("product", "金属零件"))
    return FileResponse(pdf_path, filename=f"质检报告_{session_id}.pdf",
                        media_type="application/pdf")


def saved_report(record):
    from fastapi.responses import FileResponse
    from interface.report import generate_pdf
    from review.workflow import digest
    path = os.path.join(RESULTS_DIR, "reports", digest(record)[:24] + ".pdf")
    generate_pdf(record, path, sample_name=record.get("sample_name", "未命名工件"))
    return FileResponse(path, filename="质检报告.pdf", media_type="application/pdf")


@app.get("/api/history/report")
async def history_report(day: str, key: str):
    _, record = workflow_call(get_review_workflow().locate, day=day, key=key)
    return saved_report(record)


@app.get("/api/review/queue")
async def review_queue(product: str = "", status: str = "all", offset: int = 0, limit: int = 50):
    if status not in ("all", "pending", "reviewed", "errors", "flagged", "sampled", "gray", "automatic", "legacy") or offset < 0 or not 1 <= limit <= 100:
        raise HTTPException(400, "复核筛选或分页参数无效")
    return get_review_workflow().queue(product, status, offset, limit)


@app.post("/api/review/confirm")
async def review_confirm(request: dict):
    if not str(request.get("reviewer", "")).strip():
        raise HTTPException(400, "请填写复核人员")
    return workflow_call(get_review_workflow().confirm, request)


@app.get("/api/review/record")
async def review_record(session_id: str):
    path, _ = workflow_call(get_review_workflow().locate, session_id=session_id)
    return {"ok": True, "day": path.parent.name, "key": path.stem}


@app.get("/api/iterations")
async def iteration_list():
    return workflow_call(get_review_workflow().list_runs)


@app.post("/api/iterations")
async def iteration_start(request: dict):
    return workflow_call(get_review_workflow().start, request.get("product", ""), request.get("mode", "model"))


@app.post("/api/iterations/{run_id}/activate")
async def iteration_activate(run_id: str):
    with _inference_lock:
        if get_batch_service().active(): raise HTTPException(409, "请等待后台批次结束后发布模型")
        result = workflow_call(get_review_workflow().activate, run_id)
        reset_product_cache()
        return result


@app.post("/api/iterations/{run_id}/rollback")
async def iteration_rollback(run_id: str):
    with _inference_lock:
        if get_batch_service().active(): raise HTTPException(409, "请等待后台批次结束后回滚模型")
        result = workflow_call(get_review_workflow().rollback, run_id)
        reset_product_cache()
        return result

@app.post("/api/feedback")
async def feedback(req_data: dict):
    """Record an opinion; only final human review can approve a training label."""
    from interface.report import save_feedback, get_feedback_summary
    label = req_data.get("label")
    if label not in ("up", "down"):
        raise HTTPException(400, "请选择有效的反馈选项")
    session = req_data.get("session_id", "")
    if not session:
        raise HTTPException(400, "缺少检测会话标识")
    workflow = get_review_workflow()
    with workflow.lock:
        path, _ = workflow_call(workflow.locate, session_id=session)
        saved = save_feedback(str(path), {"label": label, "comment": str(req_data.get("comment", ""))[:1000],
                                         "message_idx": 0, "pending_confirm": label == "down"})
        from review.disposition import effective_disposition
        from review.workflow import read
        disposition = effective_disposition(read(path))
    return JSONResponse({"ok": True, "applied": True, "learned": False, "model_changed": False,
                         "candidate_saved": False, "needs_review": saved.get("needs_review", False),
                         "disposition": disposition,
                         "note": "反馈已记录；人工确认真实标签并批准后才可用于候选迭代，当前模型和阈值未改变",
                         "summary": get_feedback_summary()})


@app.post("/api/confirm")
async def confirm(req_data: dict):
    """人工标签进入独立候选池；线上正常记忆库始终不变。
    body: {session_id, verdict: "ok"|"defect", defect_type: str}"""
    return workflow_call(get_review_workflow().confirm, req_data)

@app.get("/api/feedback/summary")
async def feedback_summary():
    """反馈统计与阈值调整建议"""
    from interface.report import get_feedback_summary
    return JSONResponse(get_feedback_summary())

# 挂载静态文件
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            html = f.read()
        if PUBLIC_REQUEST.get():
            html = html.replace("</head>", '<script src="/static/public-demo.js?v=20261025" defer></script></head>')
        elif os.environ.get("PUBLIC_DEMO", "0") == "1" and os.environ.get("CLOUD_ALLOW_TRAINING", "0") != "1":
            html = html.replace("</head>", '<script>window.cloudTrainingDisabled=true;</script><script src="/static/cloud-admin.js?v=20261025" defer></script></head>')
        if request.scope.get("demo_authenticated"):
            html = html.replace("</head>", '<script src="/static/session-ui.js?v=20261025" defer></script></head>')
        return html
    return "<h1>API is running. Add static/index.html</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
