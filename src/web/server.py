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

        # 2. 门控（使用该产品的独立阈值）
        g = gate(score, threshold=prod_threshold)

        # 3. 认知层
        cognition = None
        if g["route"] == "slow":
            kb = get_kb()
            hint = f"异常分数{score:.4f}"
            retrieved = kb.retrieve(hint, top_k=3)
            texts = [f"[{r['id']}] P{r['page']} {r['section']}: {r['text']}" for r in retrieved]
            prompt = vlm_api.build_prompt(hint, texts)
            tmp_path = os.path.join(RESULTS_DIR, "perception", "_api_tmp.png")
            img.save(tmp_path)
            cognition = vlm_api.analyze(tmp_path, prompt)
        else:
            if score < prod_threshold:
                cognition = {
                    "ok": True,
                    "conclusion": "合格",
                    "defect_desc": "未发现明显异常",
                    "suggestion": "正常放行",
                }
            else:
                # 快链路超阈值：根据CLIP分类给出缺陷类型
                top_defect = "未知"
                if classification:
                    top_defect = max(classification, key=classification.get)
                cognition = {
                    "ok": True,
                    "conclusion": "不合格",
                    "defect_desc": f"检测到异常区域，疑似{top_defect}缺陷" if top_defect != "正常" else "检测到局部异常，建议进一步分析",
                    "suggestion": "建议人工复核确认",
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

@app.get("/api/stats")
async def stats():
    """返回统计数据"""
    from interface import report
    records = report.list_history(limit=500)
    total = len(records)
    today = datetime.now().strftime("%Y-%m-%d")
    today_recs = [r for r in records if r.get("timestamp","").startswith(today)]
    fail = sum(1 for r in records if (r.get("cognition") or {}).get("conclusion") == "不合格")
    fast = sum(1 for r in records if r.get("gate",{}).get("route") == "fast")
    return JSONResponse({
        "total": total,
        "today": len(today_recs),
        "fail": fail,
        "pass": total - fail,
        "fast": fast,
        "slow": total - fast,
        "threshold": THRESHOLD,
    })

@app.post("/api/chat")
async def chat(req_data: dict):
    """AI助手问答"""
    import requests as http_req
    from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DASHSCOPE_MODEL
    from interface import report as rep
    question = req_data.get("question", "")
    records = rep.list_history(limit=50)
    total = len(records)
    today = datetime.now().strftime("%Y-%m-%d")
    today_recs = [r for r in records if r.get("timestamp","").startswith(today)]
    fail = sum(1 for r in records if (r.get("cognition") or {}).get("conclusion") == "不合格")
    sys_p = f"""你是智能质检系统的管理助手。基于以下数据回答，简洁专业。
    历史统计：共{total}条，今日{len(today_recs)}条，不合格{fail}条。
    判定阈值：{THRESHOLD}。
    系统功能：单图检测、批量检测、AI缺陷分析、行业标准对照。
    """
    payload = {
        "model": DASHSCOPE_MODEL,
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": question}
        ],
        "temperature": 0.3,
        "stream": False,
    }
    try:
        r = http_req.post(f"{DASHSCOPE_BASE_URL}/chat/completions",
                     headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"},
                     json=payload, timeout=30)
        d = r.json()
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
    """三级反馈机制：
    一级：人工复核结果存入案例库
    二级：案例库统计辅助调整阈值
    三级：标准库更新需人工审核
    """
    from interface.report import save_feedback, get_feedback_summary
    session_id = req_data.get("session_id", "")
    label = req_data.get("label", "")  # up/down
    comment = req_data.get("comment", "")
    tmp_dir = os.path.join(RESULTS_DIR, "perception", "web_tmp")
    # 找到对应的历史记录（最新的）
    history_dir = os.path.join(RESULTS_DIR, "history")
    today = datetime.now().strftime("%Y-%m-%d")
    day_dir = os.path.join(history_dir, today)
    if os.path.isdir(day_dir):
        files = sorted([f for f in os.listdir(day_dir) if f.endswith(".json")], reverse=True)
        if files:
            history_path = os.path.join(day_dir, files[0])
            save_feedback(history_path, {"label": label, "comment": comment, "message_idx": 0})
    summary = get_feedback_summary()
    return JSONResponse({"ok": True, "summary": summary})

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
