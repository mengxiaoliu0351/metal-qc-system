# -*- coding: utf-8 -*-
"""历史记录页"""
import os
import streamlit as st
from interface.common import CSS, HERO_HTML, sidebar, THRESHOLD, report

st.set_page_config(page_title="历史记录", page_icon="📜", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)
sidebar()

st.markdown("## 📜 检测历史")

all_hist = report.list_history(limit=50)
if not all_hist:
    st.info("暂无历史记录，去「单图检测」跑一次吧。")
else:
    options = [f"{h.get('timestamp','')[:16].replace('T',' ')} | {h.get('sample_name','')}" for h in all_hist]
    pick = st.selectbox("选择一条查看详情", options)
    idx = options.index(pick)
    h = all_hist[idx]

    st.markdown(f'<div class="card"><h3>{h.get("sample_name","")} '
                f'<small style="color:#888;font-weight:normal">{h.get("timestamp","")[:16].replace("T"," ")}</small></h3>',
                unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("异常分数", f"{h.get('score',0):.4f}", f"阈值 {THRESHOLD:.4f}")
    g = h.get("gate", {})
    pc2.metric("路由", "快链路 ⚡" if g.get("route")=="fast" else "慢链路 🔍", g.get("label",""))
    cog = h.get("cognition", {})
    pc3.metric("VLM结论", cog.get("conclusion","—") if cog and cog.get("ok") else "—")

    if h.get("input_png") and os.path.exists(h["input_png"]):
        st.image(h["input_png"], caption="待检图像", width=350)
    if h.get("heatmap_png") and os.path.exists(h["heatmap_png"]):
        st.image(h["heatmap_png"], caption="WinCLIP 热力图", width=350)

    if cog and cog.get("ok"):
        st.markdown("**缺陷描述**：" + cog.get("defect_desc",""))
        st.markdown("**处置建议**：" + cog.get("suggestion",""))
        st.markdown("**引用标准**：" + " ".join(f'<span class="ref-tag">{r}</span>' for r in cog.get("ref_ids",[])))
        if cog.get("validation",{}).get("all_valid"):
            st.success("✅ 引用校验通过")
        else:
            st.warning("⚠️ 引用校验未通过")
    elif g.get("route") == "fast":
        st.info(f"快链路直出，分数 {h.get('score',0):.4f} ≥ 阈值。")

    fbs = h.get("feedback", [])
    if fbs:
        up = sum(1 for f in fbs if f.get("label")=="up")
        down = sum(1 for f in fbs if f.get("label")=="down")
        st.markdown(f"**人在回路反馈**：👍 {up}  👎 {down}")

    st.markdown('</div>', unsafe_allow_html=True)
