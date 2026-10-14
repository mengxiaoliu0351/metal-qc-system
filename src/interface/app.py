# -*- coding: utf-8 -*-
"""首页：系统介绍 + 一键演示 + 主动学习推荐"""
import os
import json
import streamlit as st
from interface.common import CSS, HERO_HTML, sidebar, load_kb, SAMPLE_DIR, get_threshold, RESULTS_DIR
from interface import report

st.set_page_config(page_title="双链路智能质检系统", page_icon="🔩", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)
sidebar()

kb = load_kb()

st.markdown("### 👋 欢迎使用")
st.write("左侧导航选择功能，或直接点击下方演示按钮快速体验：")

# ===== 一键演示 =====
st.markdown("#### ⚡ 一键演示（答辩专用）")
demo_map = {
    "✅ 正常件（合格）": "00_正常件.png",
    "⚡ 快链路：弯曲缺陷（高置信）": "01_弯曲缺陷_高置信.png",
    "🔍 慢链路：划痕缺陷（低置信）": "02_划痕缺陷_低置信.png",
    "🔍 慢链路：色差缺陷": "03_色差缺陷.png",
}
cols = st.columns(4)
for i, (label, fn) in enumerate(demo_map.items()):
    path = os.path.join(SAMPLE_DIR, fn)
    if cols[i].button(label, key=f"demo_{i}", use_container_width=True):
        if os.path.exists(path):
            st.session_state["_demo_image"] = path
            st.session_state["_demo_sample_name"] = fn
            st.switch_page("pages/1_单图检测.py")
        else:
            cols[i].error(f"找不到 {fn}")

# ===== 主动学习推荐 =====
st.markdown("---")
st.markdown("#### 🎯 主动学习推荐（系统最不确定的样本）")
cur_thr = get_threshold()
records = report.list_history(limit=200)
# 挑分数在阈值±0.02范围内、且还没被标注过的
uncertain = []
for r in records:
    s = float(r.get("score", 0))
    if abs(s - cur_thr) < 0.02 and not r.get("active_learning_labeled"):
        uncertain.append(r)
uncertain = uncertain[:3]

if not uncertain:
    st.success("✅ 暂无待确认样本，系统对当前检测结果较有信心。")
else:
    st.write(f"以下 **{len(uncertain)}** 张图系统最不确定（分数接近阈值 {cur_thr:.4f}），请人工确认：")
    for i, r in enumerate(uncertain):
        with st.container():
            cc1, cc2, cc3 = st.columns([1, 2, 2])
            img = r.get("input_png")
            if img and os.path.exists(img):
                cc1.image(img, width=120)
            cc2.metric("异常分数", f"{r.get('score',0):.4f}", f"距阈值 {r.get('score',0)-cur_thr:+.4f}")
            cc2.caption(f"样品：{r.get('sample_name','')}")
            if cc3.button("✅ 合格", key=f"al_ok_{i}", use_container_width=True):
                report.label_active_learning(r, "合格")
                st.rerun()
            if cc3.button("❌ 不合格", key=f"al_bad_{i}", use_container_width=True):
                report.label_active_learning(r, "不合格")
                st.rerun()

st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="card"><h3>🔍 单图检测</h3>'
                '<p>选一张图 → 感知层打分 → 门控路由 → 慢链路深度分析 → 导出PDF报告</p></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card"><h3>📦 批量检测</h3>'
                '<p>一次上传多张图 → 自动逐张检测 → 汇总表格 → 导出批量PDF</p></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card"><h3>📜 历史记录</h3>'
                '<p>查看所有历史检测结果，可展开完整详情（图、热力图、VLM结论）</p></div>',
                unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🔬 系统能力")
st.write("""
- **智能感知**：自动识别金属表面异常区域，生成热力图定位
- **智能分级**：高置信结果快速直出，低置信结果深度分析
- **AI辅助**：自动描述缺陷类型，对照行业标准给出判定建议
- **人在回路**：所有结果为辅助判断，最终由人工确认签发
- **主动学习**：系统主动推荐不确定样本，持续优化检测精度
""")
