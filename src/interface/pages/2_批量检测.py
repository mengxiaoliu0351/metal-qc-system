# -*- coding: utf-8 -*-
"""批量检测页"""
import os
import datetime
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from interface.common import (CSS, HERO_HTML, sidebar, load_kb, run_perception,
                               run_cognition, SAMPLE_DIR, RESULTS_DIR, gate, cache, report)

st.set_page_config(page_title="批量检测", page_icon="📦", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)
sidebar()

kb = load_kb()

st.markdown("## 📦 批量检测")
st.markdown('<div class="card"><h3>上传多图</h3>', unsafe_allow_html=True)
batch_files = st.file_uploader("上传多张待检图像", type=["png","jpg","jpeg"], accept_multiple_files=True)
use_samples = st.checkbox("同时检测内置样例", value=False)

if st.button("▶️ 批量开始检测", type="primary"):
    items = []
    if use_samples and os.path.exists(SAMPLE_DIR):
        for f in os.listdir(SAMPLE_DIR):
            if f.lower().endswith((".png",".jpg",".jpeg")):
                items.append((os.path.join(SAMPLE_DIR, f), f))
    if batch_files:
        for uf in batch_files:
            tmp = os.path.join(RESULTS_DIR, "perception", f"_batch_{uf.name}")
            with open(tmp, "wb") as f:
                f.write(uf.getbuffer())
            items.append((tmp, uf.name))
    if not items:
        st.warning("请先上传图像或勾选内置样例")
    else:
        progress = st.progress(0, text="批量检测中...")
        batch_results = []
        for i, (ipath, iname) in enumerate(items):
            cached = cache.load_cached(ipath)
            if cached:
                r = cached
            else:
                perc = run_perception(ipath)
                if perc is None:
                    r = {"score": 0.3, "heatmap_png": None, "input_png": ipath}
                else:
                    r = perc
                g = gate(r.get("score", 0))
                r["gate"] = g
                r["input_png"] = ipath
                if g["route"] == "slow":
                    retrieved = kb.retrieve(f"异常分数{g['score']:.4f}", top_k=3)
                    r["cognition"] = run_cognition(kb, ipath, f"异常分数{g['score']:.4f}", retrieved)
                cache.save_cache(ipath, r)
            report.save_history(r, iname)
            batch_results.append((iname, r))
            progress.progress((i+1)/len(items), text=f"[{i+1}/{len(items)}] {iname}")
        progress.empty()

        st.markdown("### 📋 汇总表")
        rows = []
        for iname, r in batch_results:
            g = r.get("gate", {})
            cog = r.get("cognition", {})
            verdict = cog.get("conclusion","—") if cog and cog.get("ok") else "快链路"
            rows.append({"样品": iname, "异常分数": f"{r.get('score',0):.4f}",
                         "路由": g.get("route","—"), "VLM结论": verdict})
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Excel导出
        import pandas as pd
        df_out = pd.DataFrame(rows)
        excel_path = os.path.join(RESULTS_DIR, "reports",
                                   f"批量报告_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)
        df_out.to_excel(excel_path, index=False, sheet_name="检测结果")
        with open(excel_path, "rb") as f:
            st.download_button("📊 导出Excel", f, file_name=os.path.basename(excel_path),
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # 批量PDF
        batch_pdf = os.path.join(RESULTS_DIR, "reports",
                                 f"批量报告_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        os.makedirs(os.path.dirname(batch_pdf), exist_ok=True)
        doc = SimpleDocTemplate(batch_pdf, pagesize=A4)
        s = getSampleStyleSheet()
        elems = [Paragraph("批量质检汇总报告", s["Title"]), Spacer(1, 20)]
        for iname, r in batch_results:
            g = r.get("gate", {})
            cog = r.get("cognition", {})
            v = cog.get("conclusion","快链路直出") if cog else "快链路直出"
            elems.append(Paragraph(f"{iname}：分数{r.get('score',0):.4f} | 路由{g.get('route','—')} | 结论：{v}", s["Normal"]))
            elems.append(Spacer(1, 8))
        doc.build(elems)
        with open(batch_pdf, "rb") as f:
            st.download_button("📄 导出批量汇总PDF", f, file_name=os.path.basename(batch_pdf), mime="application/pdf")
st.markdown('</div>', unsafe_allow_html=True)
