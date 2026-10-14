# -*- coding: utf-8 -*-
"""统计看板页：产线管理者视角的数据汇总"""
import os, sys
from datetime import datetime, defaultdict
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from interface.common import CSS, HERO_HTML, sidebar, RESULTS_DIR, get_threshold
from interface import report

st.set_page_config(page_title="统计看板", page_icon="📊", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)
sidebar()

st.markdown("## 📊 检测统计看板")

# 读取所有历史
records = report.list_history(limit=500)
if not records:
    st.info("暂无检测记录。")
    st.stop()

# 构造DataFrame
rows = []
for h in records:
    ts = h.get("timestamp", "")
    date = ts[:10] if ts else "unknown"
    cog = h.get("cognition") or {}
    gate = h.get("gate") or {}
    fbs = h.get("feedback") or []
    up = sum(1 for f in fbs if f.get("label") == "up")
    down = sum(1 for f in fbs if f.get("label") == "down")
    rows.append({
        "date": date,
        "time": ts[11:16] if len(ts) > 16 else "",
        "sample": h.get("sample_name", ""),
        "score": float(h.get("score", 0)),
        "route": "快链路" if gate.get("route") == "fast" else "慢链路",
        "conclusion": cog.get("conclusion", "快链路直出"),
        "defect_type": cog.get("defect_type", "未分类"),
        "up": up, "down": down,
    })
df = pd.DataFrame(rows)

# KPI卡片
c1, c2, c3, c4 = st.columns(4)
total = len(df)
fail = len(df[df["conclusion"].str.contains("不合格", na=False)])
fast = len(df[df["route"] == "快链路"])
fb = df["up"].sum() + df["down"].sum()
c1.metric("总检测数", total)
c2.metric("不合格数", fail, f"合格率 {(total-fail)/total*100:.1f}%")
c3.metric("快链路比例", f"{fast/total*100:.0f}%", f"VLM调用 {(total-fast)/total*100:.0f}%")
c4.metric("人在回路反馈", int(fb), f"👍{int(df['up'].sum())} 👎{int(df['down'].sum())}")

st.markdown("---")

# 左：按日期趋势；右：链路分布
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("### 每日检测量与合格率")
    daily = df.groupby("date").agg(
        total=("sample", "count"),
        fail=("conclusion", lambda x: sum(1 for v in x if "不合格" in str(v)))
    ).reset_index()
    daily["pass_rate"] = (daily["total"] - daily["fail"]) / daily["total"] * 100
    fig = go.Figure()
    fig.add_bar(x=daily["date"], y=daily["total"], name="检测数", marker_color="#2c5364")
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["pass_rate"], name="合格率%",
                             yaxis="y2", line=dict(color="#e53935", width=3)))
    fig.update_layout(yaxis=dict(title="检测数"), yaxis2=dict(title="合格率%", overlaying="y", side="right", range=[0,100]),
                      height=350, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.markdown("### 快/慢链路分布")
    fig2 = px.pie(df, names="route", color="route",
                  color_discrete_map={"快链路": "#43a047", "慢链路": "#ff9800"},
                  hole=0.4)
    fig2.update_layout(height=350, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig2, use_container_width=True)

# 下：分数分布 + 缺陷类型
col_bl, col_br = st.columns(2)
with col_bl:
    st.markdown("### 异常分数分布（含阈值线）")
    cur_thr = get_threshold()
    fig3 = px.histogram(df, x="score", nbins=20, color_discrete_sequence=["#2c5364"])
    fig3.add_vline(x=cur_thr, line_dash="dash", line_color="red",
                   annotation_text=f"阈值{cur_thr:.4f}")
    fig3.update_layout(height=350, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig3, use_container_width=True)

with col_br:
    st.markdown("### 缺陷类型分布")
    fail_df = df[df["conclusion"].str.contains("不合格", na=False)]
    if len(fail_df) == 0:
        st.info("暂无不合格记录。")
    else:
        dt_counts = fail_df["defect_type"].value_counts().reset_index()
        dt_counts.columns = ["defect_type", "count"]
        fig4 = px.pie(dt_counts, names="defect_type", values="count",
                      hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
        fig4.update_layout(height=350, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig4, use_container_width=True)

# 反馈统计
st.markdown("### 反馈统计")
fb_df = df[df["up"] + df["down"] > 0][["sample", "conclusion", "defect_type", "up", "down"]]
if len(fb_df) == 0:
    st.info("暂无反馈数据。在检测页点👍/👎后会显示在这里。")
else:
    st.dataframe(fb_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("数据来源：results/history/ 目录下的检测记录JSON")
