# -*- coding: utf-8 -*-
"""共享资源：模型、检测函数、样式、侧边栏"""
import os, sys
import numpy as np
import streamlit as st
from PIL import Image

ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "perception"))

from config import THRESHOLD, VLM_BACKEND, SAMPLE_DIR, RESULTS_DIR
from gating.router import gate
from cognition.rag import KnowledgeBase
from cognition import vlm_api
from interface import cache, report

# ---------- 反馈自动调阈值 ----------
_ADJ_FILE = os.path.join(RESULTS_DIR, "perception", "adjusted_threshold.json")

def get_threshold():
    """返回当前阈值：优先用反馈调整后的值，否则用config默认值"""
    if os.path.exists(_ADJ_FILE):
        try:
            import json
            with open(_ADJ_FILE, encoding="utf-8") as f:
                d = json.load(f)
            return float(d.get("threshold", THRESHOLD))
        except Exception:
            pass
    return THRESHOLD

def maybe_adjust_threshold(feedback_label: str, score: float):
    """根据反馈自动调整阈值。
    - 低分区(分数<当前阈值)被👎 → 阈值上调10%（误判为正常，实际有缺陷）
    - 高分区(分数≥当前阈值)被👎 → 阈值下调10%（误判为缺陷，实际正常）
    每次调整幅度小，避免震荡。
    """
    cur = get_threshold()
    new = cur
    if feedback_label == "down":
        if score < cur:
            new = cur * 1.10
        else:
            new = cur * 0.90
    if abs(new - cur) < 1e-6:
        return cur, False
    os.makedirs(os.path.dirname(_ADJ_FILE), exist_ok=True)
    import json
    with open(_ADJ_FILE, "w", encoding="utf-8") as f:
        json.dump({"threshold": round(new, 4), "prev": cur,
                   "reason": f"反馈{feedback_label}, score={score:.4f}"}, f, ensure_ascii=False, indent=2)
    return new, True

CSS = """
<style>
  /* 全局深色工业主题 */
  .stApp {
    background: #0a0e1a !important;
  }
  .block-container { padding-top: 1.5rem; max-width: 1280px; }

  /* Hero横幅 */
  .hero {
      background: linear-gradient(135deg, #0a1628 0%, #0d2847 50%, #1a3a5c 100%);
      padding: 28px 36px; border-radius: 16px; color: white;
      margin-bottom: 24px;
      border: 1px solid rgba(0, 200, 255, 0.2);
      box-shadow: 0 8px 32px rgba(0, 200, 255, 0.1), inset 0 1px 0 rgba(255,255,255,0.05);
      position: relative; overflow: hidden;
  }
  .hero::before {
      content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
      background: linear-gradient(90deg, transparent, #00c8ff, transparent);
  }
  .hero h1 { margin: 0; font-size: 24px; font-weight: 700; letter-spacing: 0.5px; }
  .hero .sub { opacity: 0.7; font-size: 13px; margin-top: 8px; }
  .hero .badge {
      background: rgba(0, 200, 255, 0.15); padding: 8px 16px; border-radius: 20px;
      font-size: 12px; border: 1px solid rgba(0, 200, 255, 0.3);
  }

  /* 侧边栏 */
  section[data-testid="stSidebar"] {
      background: #0d1220 !important;
      border-right: 1px solid rgba(255,255,255,0.05);
  }
  section[data-testid="stSidebar"] * { color: #8899bb !important; }
  section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
      color: #00c8ff !important; font-size: 16px !important;
  }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stCheckbox label { color: #aac !important; }

  /* 卡片 */
  .card {
      background: rgba(255,255,255,0.03); padding: 24px; border-radius: 14px;
      margin-bottom: 20px;
      border: 1px solid rgba(255,255,255,0.08);
      backdrop-filter: blur(10px);
  }
  .card h3 { margin-top: 0; color: #00c8ff; font-size: 16px; font-weight: 600; }
  .card p { color: #8899bb; font-size: 13px; line-height: 1.6; }

  /* 指标卡 */
  div[data-testid="stMetric"] {
      background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.08);
      border-left: 3px solid #00c8ff;
  }
  div[data-testid="stMetric"] label { color: #8899bb !important; font-size: 12px !important; }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
      color: white !important; font-size: 28px !important; font-weight: 700 !important;
  }
  div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 12px !important; }

  /* 按钮 */
  div.stButton > button[kind="primary"] {
      background: linear-gradient(90deg, #00c8ff, #0099cc);
      border: none; padding: 0.6rem 2rem; font-weight: 600; border-radius: 10px;
      box-shadow: 0 4px 15px rgba(0,200,255,0.3);
      transition: all 0.3s;
  }
  div.stButton > button[kind="primary"]:hover {
      box-shadow: 0 6px 20px rgba(0,200,255,0.5);
      transform: translateY(-1px);
  }
  div.stButton > button {
      background: rgba(255,255,255,0.05); color: #ccd; border: 1px solid rgba(255,255,255,0.1);
      border-radius: 10px;
  }

  /* 判定结果 */
  .verdict-fail {
      background: rgba(229,57,53,0.1);
      border-left: 4px solid #e53935; padding: 16px 20px; border-radius: 10px; margin: 12px 0;
  }
  .verdict-pass {
      background: rgba(67,160,71,0.1);
      border-left: 4px solid #43a047; padding: 16px 20px; border-radius: 10px; margin: 12px 0;
  }

  /* 引用标签 */
  .ref-tag {
      display: inline-block; background: rgba(0,200,255,0.1); color: #00c8ff;
      padding: 4px 14px; border-radius: 14px; font-family: monospace; font-size: 12px;
      margin: 2px 4px 2px 0; border: 1px solid rgba(0,200,255,0.3);
  }

  /* 分隔线 */
  hr { border-color: rgba(255,255,255,0.08) !important; }

  /* 文本颜色 */
  h1, h2, h3 { color: #e0e8f0 !important; }
  p, li { color: #a0b0c0 !important; }
  .stMarkdown { color: #a0b0c0; }

  /* 输入框 */
  .stTextInput input, .stTextArea textarea {
      background: rgba(255,255,255,0.05) !important; color: white !important;
      border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 8px !important;
  }

  /* 表格 */
  .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
"""

HERO_HTML = """
<div class="hero">
  <div>
    <h1>🔩 金属表面缺陷 · 双链路智能质检辅助系统</h1>
    <div class="sub">智能图像识别 + AI缺陷分析 + 行业标准对照 · 人在回路辅助决策</div>
  </div>
  <div class="badge">山东大学人工智能学院</div>
</div>
"""

@st.cache_resource
def load_kb():
    return KnowledgeBase(csv_path=os.path.join(RESULTS_DIR, "cognition", "standard_library.csv"))

@st.cache_resource
def load_engine(product: str = "金属螺母(metal_nut)"):
    try:
        from perception.winclip_patch import WinCLIPPatch
        det = WinCLIPPatch("ViT-B-16")
        banks_dir = os.path.join(RESULTS_DIR, "perception", "banks")
        os.makedirs(banks_dir, exist_ok=True)
        bank_cache = os.path.join(banks_dir, f"{product}.npy")
        if os.path.exists(bank_cache):
            det.bank = np.load(bank_cache)
            det.bank_info = {"n_images": "?", "n_feats": det.bank.shape[0], "product": product}
        elif product == "金属螺母(metal_nut)":
            # 首次启动用metal_nut建库
            det.build_bank(
                os.path.join(ROOT, "data", "mvtec", "metal_nut", "train", "good"),
                max_samples=80, cache_path=bank_cache)
        else:
            return None
        return det
    except Exception as e:
        st.warning(f"感知层模型加载失败（{e}）")
        return None

def list_products():
    """列出所有已建库产品"""
    banks_dir = os.path.join(RESULTS_DIR, "perception", "banks")
    os.makedirs(banks_dir, exist_ok=True)
    prods = ["金属螺母(metal_nut)"]
    for f in os.listdir(banks_dir):
        if f.endswith(".npy"):
            name = f[:-4]
            if name != "金属螺母(metal_nut)":
                prods.append(name)
    return prods

def build_bank_for_product(product: str, image_paths: list):
    """为新产品建记忆库"""
    import numpy as np
    from perception.winclip_patch import WinCLIPPatch
    det = WinCLIPPatch("ViT-B-16")
    feats = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        feats.append(det._get_patch_feats(img))
    det.bank = np.concatenate(feats, axis=0)
    banks_dir = os.path.join(RESULTS_DIR, "perception", "banks")
    os.makedirs(banks_dir, exist_ok=True)
    np.save(os.path.join(banks_dir, f"{product}.npy"), det.bank)
    return {"n_images": len(image_paths), "n_feats": det.bank.shape[0]}

def run_perception(image_path, product: str = None):
    if product is None:
        product = st.session_state.get("_product", "金属螺母(metal_nut)")
    det = load_engine(product)
    if det is None:
        return None
    try:
        image = Image.open(image_path).convert("RGB")
        r = det.detect(image)
        input_png = os.path.join(RESULTS_DIR, "perception", "_live_input.png")
        heat_png = os.path.join(RESULTS_DIR, "perception", "_live_heatmap.png")
        image.save(input_png)
        r["heatmap"].save(heat_png)
        return {"score": float(r["score"]), "heatmap_png": heat_png, "input_png": input_png,
                "classification": r.get("classification", {})}
    except Exception as e:
        st.error(f"感知推理失败: {e}")
        return None

def run_cognition(kb_obj, crop_png, defect_hint, retrieved, score=None, threshold=None):
    texts = [f"[{r['id']}] P{r['page']} {r['section']}: {r['text']}" for r in retrieved]
    prompt = vlm_api.build_prompt(defect_hint, texts)
    out = vlm_api.analyze(crop_png, prompt,
                          backend="local" if VLM_BACKEND == "local" else "api")
    if not out.get("ok"):
        # 降级：VLM挂了不崩溃，基于WinCLIP分数给保守结论
        s = score if score is not None else 0
        t = threshold if threshold is not None else THRESHOLD
        danger = "异常分数高，建议人工复核" if s >= t else "异常分数低，VLM暂不可用，建议人工确认"
        return {
            "ok": True,  # 整体流程不失败
            "degraded": True,
            "conclusion": "需人工复核",
            "defect_desc": f"VLM慢链路暂不可用（{out.get('error','')}）。{danger}。",
            "ref_ids": [],
            "suggestion": "请人工查看热力图并签发结论。",
            "raw": out.get("raw", ""),
            "validation": {"all_valid": False},
            "retrieved": retrieved,
        }
    valid = kb_obj.validate_ref_ids(out["ref_ids"], [r["id"] for r in retrieved])
    out["validation"] = valid
    out["retrieved"] = retrieved
    return out

def sidebar():
    st.sidebar.markdown("## 🎛️ 系统状态")
    st.sidebar.markdown("---")
    # 操作员选择
    operators = ["张伟", "李芳", "王强", "赵静", "（自定义）"]
    op = st.sidebar.selectbox("操作员", operators, index=0)
    if op == "（自定义）":
        op = st.sidebar.text_input("输入操作员姓名", "") or "未署名"
    st.session_state["_operator"] = op
    # 产品选择器
    prods = list_products()
    cur = st.sidebar.selectbox("当前检测产品", prods,
                                index=prods.index(st.session_state.get("_product", prods[0]))
                                if st.session_state.get("_product", prods[0]) in prods else 0)
    st.session_state["_product"] = cur
    cur_thr = get_threshold()
    st.sidebar.caption(f"门控阈值：{cur_thr}")
    st.sidebar.caption(f"操作员：{op}")
    st.sidebar.caption("VLM：Qwen3.6-VL 云端API")
    st.sidebar.caption("感知：WinCLIP (CLIP ViT-B-16)")
    st.sidebar.caption("知识库：GB/T 34528-2017")
