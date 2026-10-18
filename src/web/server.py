# -*- coding: utf-8 -*-
"""FastAPI后端：包装检测逻辑，提供REST接口给HTML前端"""
import os, sys, io, json, base64, time
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import THRESHOLD, RESULTS_DIR
from gating.router import gate
from cognition.rag import KnowledgeBase
from cognition import vlm_api

app = FastAPI(title="智能质检系统API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 全局加载模型（启动时加载一次）
_engine = None
_kb = None
_sem_kb = None
_clfs = {}
_products = None
_current_product = None

BANKS_DIR = os.path.join(RESULTS_DIR, "perception", "banks")


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
    if _engine is None:
        from perception.winclip_patch import WinCLIPPatch
        _engine = WinCLIPPatch("ViT-B-16")

    products = load_products()
    pcfg = next((p for p in products if p["key"] == product_key), products[0])

    if _current_product != product_key:
        bank_path = os.path.join(BANKS_DIR, pcfg["bank"])
        _engine.bank = np.load(bank_path)
        _engine.bank_info = {"n_images": 80, "n_feats": _engine.bank.shape[0]}
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
    """#3 CLIP 语义国标检索：复用引擎的文本编码器，条文一次性编码缓存"""
    global _sem_kb
    if _sem_kb is None:
        import torch, open_clip
        from cognition.rag import SemanticKnowledgeBase
        def encode(texts):
            tok = open_clip.tokenize(texts).to(engine.device)
            with torch.no_grad():
                f = engine.model.encode_text(tok)
            return f.cpu().numpy()
        _sem_kb = SemanticKnowledgeBase(encode)
    return _sem_kb

def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@app.get("/api/products")
async def list_products():
    """列出可用产品记忆库 + VLM通用兜底模式"""
    products = load_products()
    return JSONResponse({
        "products": [{"key": p["key"], "name": p["name"],
                      "threshold": p["threshold"]} for p in products],
        "fallback": {"key": "vlm_general",
                     "name": "通用直检（未建库产品 · AI视觉兜底）"},
    })


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), product: str = Form("metal_nut")):
    """检测接口：上传图片，返回异常分数、热力图、缺陷类型、VLM分析。
    product=vlm_general 时走VLM通用直检兜底（不依赖记忆库）。"""
    try:
        # 读图片
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")

        # ===== VLM 通用直检兜底（未建库新产品）=====
        if product == "vlm_general":
            t0 = time.time()
            tmp_path = os.path.join(RESULTS_DIR, "perception", "_api_tmp.png")
            img.save(tmp_path)
            prompt = ("该产品尚未建立正常样本库，请直接基于图像本身判断金属表面是否存在可见缺陷"
                      "（划痕、裂纹、凹坑、锈斑、变形、色差等）。按系统规定的JSON格式输出结论，"
                      "ref_ids 若无对应标准可留空数组，suggestion 给出处置建议。")
            cog = vlm_api.analyze(tmp_path, prompt)
            if not cog.get("ok"):
                cog = {"ok": True, "conclusion": "需复核",
                       "defect_desc": cog.get("error", "AI直检暂不可用，请人工判读"),
                       "suggestion": "建议人工复核"}
            t_vlm = time.time() - t0
            verdict = cog.get("conclusion") or "需复核"
            import uuid
            session_id = datetime.now().strftime("%H%M%S_") + uuid.uuid4().hex[:6]
            tmp_dir = os.path.join(RESULTS_DIR, "perception", "web_tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            img.save(os.path.join(tmp_dir, f"{session_id}_input.png"))
            return JSONResponse({
                "ok": True, "session_id": session_id,
                "score": None, "threshold": None,
                "route": "general", "route_label": "通用直检 · AI视觉分析",
                "heatmap": None, "classification": None,
                "cognition": {
                    "ok": True,
                    "conclusion": verdict,
                    "defect_desc": cog.get("defect_desc") or cog.get("reason") or "见AI分析",
                    "suggestion": cog.get("suggestion") or "建议人工复核",
                    "general": True,
                },
                "time_perception": round(t_vlm, 2),
            })

        # ===== 常规双链路：按产品加载对应记忆库与阈值 =====
        engine, pcfg = get_engine(product)
        prod_threshold = float(pcfg["threshold"])
        t0 = time.time()
        result = engine.detect(img)
        t_perception = time.time() - t0

        score = float(result["score"])
        heatmap_b64 = img_to_b64(result["heatmap"])
        classification = result.get("classification", {})

        # #1 监督缺陷类型分类（各产品独立训练，有模型才跑）
        sup_clf = None
        try:
            clf = get_clf(product)
            if clf.available:
                sup_clf = clf.predict(engine, img)
        except Exception as _e:
            print("监督分类失败:", _e)

        # 2. 三区间门控（低风险/灰区VLM/高风险）
        cal = pcfg.get("calibration", {}) or {}
        _med = float(cal.get("median", 0) or 0)
        _mad = float(cal.get("mad", 0) or 0)
        # t_high优先级: products.json显式配置 > median+3*MAD > t_low*1.5
        t_high = float(pcfg.get("t_high", 0) or 0)
        if t_high <= 0 and _med > 0 and _mad > 0:
            t_high = round(_med + 3.0 * _mad, 4)
        if t_high <= 0:
            t_high = round(prod_threshold * 1.5, 4)
        g = gate(score, t_low=prod_threshold, t_high=t_high)

        # 3. 认知层：仅灰区调 VLM + 国标 RAG；低/高风险本地直出，省 API 调用
        cognition = None
        if g["route"] == "vlm_review":
            # 灰区：调 VLM 深度复核 + RAG
            defect_word = "表面缺陷"
            if sup_clf and sup_clf.get("label") and sup_clf["label"] != "正常":
                defect_word = sup_clf["label"]
            elif classification:
                defect_word = max(classification, key=classification.get)
            query = f"{defect_word}金属表面缺陷"
            sem_kb = get_sem_kb(engine)
            retrieved = sem_kb.retrieve(query, top_k=3)
            texts = [f"[{r['id']}] P{r['page']} {r['section']}: {r['text']}" for r in retrieved]
            hint = f"异常分数{score:.4f}（介于{prod_threshold}与{t_high}之间），疑似{defect_word}缺陷"
            prompt = vlm_api.build_prompt(hint, texts)
            tmp_path = os.path.join(RESULTS_DIR, "perception", "_api_tmp.png")
            img.save(tmp_path)
            cognition = vlm_api.analyze(tmp_path, prompt)
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
                "suggestion": "分数超过高风险阈值，请人工复核确认后签发",
            }
        else:
            # 低风险：本地放行，不调 VLM
            cognition = {
                "ok": True,
                "conclusion": "合格",
                "defect_desc": "异常分数低于低风险阈值，未发现明显异常",
                "suggestion": "低风险放行，仍建议按抽检比例人工确认",
            }

        # 保存临时文件供PDF报告使用
        import uuid
        session_id = datetime.now().strftime("%H%M%S_") + uuid.uuid4().hex[:6]
        tmp_dir = os.path.join(RESULTS_DIR, "perception", "web_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        input_path = os.path.join(tmp_dir, f"{session_id}_input.png")
        heatmap_path = os.path.join(tmp_dir, f"{session_id}_heatmap.png")
        img.save(input_path)
        result["heatmap"].save(heatmap_path)
        # 保存元信息（产品、阈值）供PDF报告复用
        with open(os.path.join(tmp_dir, f"{session_id}_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"product": pcfg["name"], "product_key": product,
                       "threshold": prod_threshold, "score": score,
                       "route": g["route"], "route_label": g["label"],
                       "cognition": cognition}, f, ensure_ascii=False, indent=2)

        # ===== 持久化图片并写入检测历史（供历史详情回看）=====
        try:
            from interface import report
            day_img = os.path.join(RESULTS_DIR, "history", "images", datetime.now().strftime("%Y-%m-%d"))
            os.makedirs(day_img, exist_ok=True)
            persist_input = os.path.join(day_img, f"{session_id}_input.png")
            persist_heat = os.path.join(day_img, f"{session_id}_heatmap.png")
            img.save(persist_input)
            result["heatmap"].save(persist_heat)
            report.save_history({
                "session_id": session_id,
                "score": score,
                "gate": {"route": g["route"], "label": g["label"], "threshold": prod_threshold},
                "cognition": cognition, "classification": classification,
                "supervised_class": sup_clf,
                "input_png": persist_input, "heatmap_png": persist_heat,
                "product_key": product, "product_name": pcfg["name"],
            }, sample_name=file.filename or "零件")
        except Exception as _e:
            print("写历史失败:", _e)

        return JSONResponse({
            "ok": True,
            "session_id": session_id,
            "product": pcfg["name"],
            "score": score,
            "threshold": prod_threshold,
            "route": g["route"],
            "route_label": g["label"],
            "heatmap": heatmap_b64,
            "classification": classification,
            "supervised_class": sup_clf,
            "cognition": cognition,
            "time_perception": round(t_perception, 2),
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})

@app.get("/api/history")
async def history(limit: int = 50):
    """返回历史记录"""
    from interface import report
    records = report.list_history(limit=limit)
    return JSONResponse({"records": records})


@app.get("/api/history/detail")
async def history_detail(day: str = "", key: str = ""):
    """回看某条历史详情：返回原图、热力图(base64)、结论等完整字段"""
    from interface import report
    day_dir = os.path.join(RESULTS_DIR, "history", day)
    p = os.path.join(day_dir, f"{key}.json")
    if not os.path.exists(p):
        raise HTTPException(404, "记录不存在或已过期")
    with open(p, encoding="utf-8") as f:
        rec = json.load(f)

    def _b64(path):
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return None

    return JSONResponse({
        "record": rec,
        "input_img": _b64(rec.get("input_png")),
        "heatmap_img": _b64(rec.get("heatmap_png")),
    })

@app.get("/api/stats")
async def stats():
    """返回统计数据"""
    from interface import report
    records = report.list_history(limit=500)
    total = len(records)
    today = datetime.now().strftime("%Y-%m-%d")
    today_recs = [r for r in records if r.get("timestamp","").startswith(today)]
    fail = sum(1 for r in records if (r.get("cognition") or {}).get("conclusion") in ("不合格", "疑似不合格"))
    low_risk = sum(1 for r in records if r.get("gate",{}).get("route") == "low_risk")
    vlm_review = sum(1 for r in records if r.get("gate",{}).get("route") == "vlm_review")
    high_risk = sum(1 for r in records if r.get("gate",{}).get("route") == "high_risk")
    products = load_products()
    real_threshold = float(products[0]["threshold"]) if products else THRESHOLD
    return JSONResponse({
        "total": total,
        "today": len(today_recs),
        "fail": fail,
        "pass": total - fail,
        "low_risk": low_risk,
        "vlm_review": vlm_review,
        "high_risk": high_risk,
        "threshold": real_threshold,
        "product_count": len(products),
    })

@app.post("/api/chat")
async def chat(req_data: dict):
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

    records = rep.list_history(limit=500)
    total = len(records)
    today = datetime.now().strftime("%Y-%m-%d")
    today_recs = [r for r in records if r.get("timestamp", "").startswith(today)]
    fail = sum(1 for r in records if (r.get("cognition") or {}).get("conclusion") == "不合格")
    pass_ = total - fail

    # 各产品阈值表
    products = load_products()
    prod_lines = "\n".join([f"- {p['name']}：判定阈值 {p['threshold']}" for p in products])

    # 按产品统计
    prod_stat = {}
    for r in records:
        pn = r.get("product_name", "未知")
        s = prod_stat.setdefault(pn, {"total": 0, "fail": 0})
        s["total"] += 1
        if (r.get("cognition") or {}).get("conclusion") == "不合格":
            s["fail"] += 1
    prod_stat_lines = "\n".join(
        f"- {k}：共{v['total']}次，不合格{v['fail']}次，合格率{round((v['total']-v['fail'])/v['total']*100,1)}%"
        for k, v in prod_stat.items()) or "（暂无记录）"

    # 按日期统计（最近7天）
    from collections import defaultdict
    day_stat = defaultdict(lambda: {"total": 0, "fail": 0})
    for r in records:
        d = (r.get("timestamp", "") or "")[:10]
        if d:
            day_stat[d]["total"] += 1
            if (r.get("cognition") or {}).get("conclusion") == "不合格":
                day_stat[d]["fail"] += 1
    day_lines = "\n".join(
        f"- {d}：共{v['total']}次，不合格{v['fail']}次"
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
3. 合格率/不合格率按给出的次数计算，保留一位小数。

【全量统计（最近500条）】
- 历史检测总数：{total}；今日：{len(today_recs)} 条；合格 {pass_} 条，不合格 {fail} 条，总合格率{round(pass_/total*100,1) if total else 0}%。

【各产品判定阈值】（异常分数≥阈值判不合格）
{prod_lines}

【各产品检测统计】
{prod_stat_lines}

【按日期统计（最近7天）】
{day_lines}

【常见缺陷类型分布】
{defect_lines}

【判据说明】快链路：分数直接对照阈值；慢链路：接近阈值时由视觉大模型复核。所有结论为辅助判断，最终人工签发。"""

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
            "暖色/高亮区代表系统判定的异常区域）。"
            + (f"\n【系统对这张图的检测结果】{det_ctx}" if det_ctx else "")
            + "\n请结合图片用中文自然语言回答：异常大概在什么位置、可能是什么缺陷类型、"
              "为什么这么判、给什么处置建议。简洁口语，不要输出JSON。")
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
    try:
        r = http_req.post(f"{DASHSCOPE_BASE_URL}/chat/completions",
                     headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"},
                     json=payload, timeout=30)
        d = r.json()
        if "choices" not in d:
            err = d.get("error", {})
            msg = err.get("message") or d.get("message") or str(d)[:200]
            hint = "（未配置或失效的大模型 API Key）" if "PASTE" in DASHSCOPE_API_KEY or "Invalid" in msg or "Authentication" in msg else ""
            return JSONResponse({"answer": f"助手暂时不可用：{msg} {hint}"})
        answer = d["choices"][0]["message"]["content"]
        return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse({"answer": f"请求失败: {str(e)}"})

@app.get("/api/report/{session_id}")
async def report_pdf(session_id: str):
    """生成PDF质检报告（读取检测时保存的元信息，使用对应产品阈值）"""
    from fastapi.responses import FileResponse
    from interface.report import generate_pdf
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
                 "threshold": meta.get("threshold", THRESHOLD)},
        "input_png": input_path,
        "heatmap_png": heatmap_path if os.path.exists(heatmap_path) else None,
        "cognition": meta.get("cognition") or {},
    }
    pdf_path = os.path.join(tmp_dir, f"{session_id}_report.pdf")
    generate_pdf(result, pdf_path, sample_name=meta.get("product", "金属零件"))
    return FileResponse(pdf_path, filename=f"质检报告_{session_id}.pdf",
                        media_type="application/pdf")

@app.post("/api/feedback")
async def feedback(req_data: dict):
    """三级反馈机制：按 session_id 精确定位到对应历史记录（不再错配最新一条）"""
    from interface.report import save_feedback, get_feedback_summary
    session_id = req_data.get("session_id", "")
    label = req_data.get("label", "")  # up/down
    comment = req_data.get("comment", "")
    history_dir = os.path.join(RESULTS_DIR, "history")

    # 按 session_id 在历史目录里定位对应 json（扫最近 7 天）
    history_path = None
    if session_id and os.path.isdir(history_dir):
        days = sorted([d for d in os.listdir(history_dir)
                       if os.path.isdir(os.path.join(history_dir, d))], reverse=True)[:7]
        for day in days:
            day_dir = os.path.join(history_dir, day)
            for fn in os.listdir(day_dir):
                if not fn.endswith(".json"):
                    continue
                p = os.path.join(day_dir, fn)
                try:
                    with open(p, encoding="utf-8") as f:
                        rec = json.load(f)
                    if rec.get("session_id") == session_id:
                        history_path = p
                        break
                except Exception:
                    continue
            if history_path:
                break

    applied = False
    learned = False
    if history_path:
        # 反馈只记录，不自动 learn()。用户点踩≠真标签，防止误判样本污染正常记忆库。
        # 必须由人工在历史详情页确认"该样本确为缺陷"后，才走 /api/confirm 回灌。
        save_feedback(history_path, {"label": label, "comment": comment, "message_idx": 0,
                                     "pending_confirm": (label == "down")})
        applied = True
    summary = get_feedback_summary()
    return JSONResponse({"ok": True, "applied": applied, "learned": learned,
                         "note": "反馈已记录，待人工确认为缺陷后才回灌记忆库", "summary": summary})


@app.post("/api/confirm")
async def confirm(req_data: dict):
    """人工确认样本：只有质检人员明确判定后才回灌记忆库，防止误污染。
    body: {session_id, verdict: "ok"|"defect", defect_type: str}"""
    session_id = req_data.get("session_id", "")
    verdict = req_data.get("verdict", "")
    defect_type = req_data.get("defect_type", "")
    history_dir = os.path.join(RESULTS_DIR, "history")

    history_path = None
    if session_id and os.path.isdir(history_dir):
        days = sorted([d for d in os.listdir(history_dir)
                       if os.path.isdir(os.path.join(history_dir, d))], reverse=True)[:7]
        for day in days:
            for fn in os.listdir(os.path.join(history_dir, day)):
                if not fn.endswith(".json"):
                    continue
                p = os.path.join(history_dir, day, fn)
                try:
                    with open(p, encoding="utf-8") as f:
                        rec = json.load(f)
                    if rec.get("session_id") == session_id:
                        history_path = p
                        break
                except Exception:
                    continue
            if history_path:
                break

    learned = False
    if history_path and verdict == "defect":
        # 人工确认是缺陷：把该图特征并入记忆库（缺陷样本回灌，提升同类检出）
        try:
            with open(history_path, encoding="utf-8") as f:
                rec = json.load(f)
            pkey = rec.get("product_key", "metal_nut")
            img_path = rec.get("input_png")
            if img_path and os.path.exists(img_path):
                engine, pcfg = get_engine(pkey)
                bank_path = os.path.join(BANKS_DIR, pcfg["bank"])
                n = engine.learn(img_path, save_path=bank_path)
                learned = True
                print(f"[人工确认回灌] {pkey} 记忆库并入已确认缺陷样本，特征数={n}")
            # 回写人工判定
            rec["human_verdict"] = {"verdict": "defect", "defect_type": defect_type,
                                    "confirmed": True, "ts": datetime.now().isoformat()}
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
        except Exception as _e:
            print("人工回灌失败:", _e)
    elif history_path:
        # 人工判为合格：回写，不进任何库
        try:
            with open(history_path, encoding="utf-8") as f:
                rec = json.load(f)
            rec["human_verdict"] = {"verdict": verdict, "confirmed": True,
                                    "ts": datetime.now().isoformat()}
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return JSONResponse({"ok": True, "learned": learned})

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
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return f.read()
    return "<h1>API is running. Add static/index.html</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
