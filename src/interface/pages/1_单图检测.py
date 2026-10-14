# -*- coding: utf-8 -*-
"""单图检测页"""
import os
import streamlit as st
from interface.common import (CSS, HERO_HTML, sidebar, load_kb, run_perception,
                               run_cognition, THRESHOLD, SAMPLE_DIR, RESULTS_DIR, gate,
                               cache, report, load_engine, build_bank_for_product,
                               get_threshold, maybe_adjust_threshold)
from cognition import vlm_api

st.set_page_config(page_title="单图检测", page_icon="🔍", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)
sidebar()

kb = load_kb()

st.markdown("## 🔍 单图检测")

# 一键演示：从首页跳转过来时自动选图
_demo_img = st.session_state.pop("_demo_image", None)
_demo_name = st.session_state.pop("_demo_sample_name", "")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('<div class="card"><h3>📥 输入图像</h3>', unsafe_allow_html=True)
    use_sample = st.checkbox("使用内置样例图", value=(_demo_img is None))
    uploaded = st.file_uploader("上传待检图像", type=["png", "jpg", "jpeg"])
    image_path = None
    sample_name = ""
    if _demo_img and os.path.exists(_demo_img):
        image_path = _demo_img
        sample_name = _demo_name
        st.success(f"已从首页载入演示图：{_demo_name}")
    elif use_sample:
        samples = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))] \
                 if os.path.exists(SAMPLE_DIR) else []
        if samples:
            chosen = st.selectbox("选择样例", samples)
            image_path = os.path.join(SAMPLE_DIR, chosen)
            sample_name = chosen
    if uploaded is not None:
        tmp = os.path.join(RESULTS_DIR, "perception", "_uploaded.png")
        with open(tmp, "wb") as f:
            f.write(uploaded.getbuffer())
        image_path = tmp
        sample_name = uploaded.name
    if image_path:
        st.image(image_path, caption="待检图像", width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><h3>🔍 检测</h3>', unsafe_allow_html=True)
    product = st.session_state.get("_product", "金属螺母(metal_nut)")
    det = load_engine(product)
    if det is None:
        st.warning(f"「{product}」还没有记忆库。请上传3-10张该产品的**正常件**图片，自动建库：")
        up_normal = st.file_uploader("上传正常件图（可多选）", type=["png","jpg","jpeg"],
                                      accept_multiple_files=True, key="_normal_upload")
        if up_normal and st.button("🏗️ 一键建库", type="primary"):
            tmp_dir = os.path.join(RESULTS_DIR, "perception", "_tmp_build")
            os.makedirs(tmp_dir, exist_ok=True)
            paths = []
            for i, f in enumerate(up_normal):
                p = os.path.join(tmp_dir, f"{i}.png")
                with open(p, "wb") as out:
                    out.write(f.getbuffer())
                paths.append(p)
            with st.spinner(f"正在为「{product}」建库（{len(paths)}张）..."):
                info = build_bank_for_product(product, paths)
            st.success(f"✅ 建库完成：{info['n_images']}张图，{info['n_feats']}个patch特征")
            st.cache_resource.clear()
            st.rerun()
        st.stop()

    st.caption("默认走智能分级检测（高置信快速出结果，低置信深度分析）。若零件未建库，可选通用AI直检。")
    use_generic = st.checkbox("🆕 通用AI直检模式（未建库零件用）", value=False)
    if image_path and st.button("▶️ 开始检测", type="primary"):
        cached = cache.load_cached(image_path)
        if cached:
            st.success("✅ 结果已就绪")
            result = cached
        else:
            result = {"input_png": image_path}
            if use_generic:
                # 通用模式：直接VLM+RAG，不经过WinCLIP
                with st.spinner("VLM通用分析中（3-8秒）..."):
                    retrieved = kb.retrieve("金属表面缺陷检测", top_k=3)
                    result["cognition"] = run_cognition(kb, image_path, "通用质检分析", retrieved)
                    result["gate"] = {"route": "slow", "label": "VLM通用模式"}
            else:
                # 主架构：WinCLIP感知 + 门控双链路
                with st.spinner("感知层推理中..."):
                    perc = run_perception(image_path)
                if perc is None:
                    st.warning("未找到该产品的记忆库。请勾选上方「VLM通用直检模式」重试，或先建库。")
                    st.stop()
                result.update(perc)
                cur_thr = get_threshold()
                g = gate(result.get("score", 0.0), threshold=cur_thr)
                result["gate"] = g
                if g["route"] == "slow":
                    with st.spinner("慢链路：VLM + RAG（3-8秒）..."):
                        hint = f"异常分数{g['score']:.4f}"
                        retrieved = kb.retrieve(hint, top_k=3)
                        result["cognition"] = run_cognition(kb, image_path, hint, retrieved)
                else:
                    # 快链路：也调一次VLM简短描述，让工人能看懂
                    with st.spinner("快链路：VLM简短描述（1-2秒）..."):
                        result["cognition"] = {
                            "ok": True,
                            "conclusion": "不合格",
                            "defect_desc": vlm_api.quick_describe(image_path),
                            "ref_ids": [],
                            "suggestion": "人工复核后签发",
                        }
            cache.save_cache(image_path, result)
        hist_path = report.save_history(result, sample_name)
        st.session_state["_last_hist_path"] = hist_path

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><h3>📊 检测结果</h3>', unsafe_allow_html=True)

        pc1, pc2, pc3 = st.columns(3)
        cur_thr = get_threshold()
        pc1.metric("异常分数", f"{result.get('score',0):.4f}", f"阈值 {cur_thr:.4f}")
        gi = result.get("gate", gate(result.get("score",0), threshold=cur_thr))
        pc2.metric("检测模式", "快速出结果 ⚡" if gi["route"]=="fast" else "深度分析 🔍", gi["label"])
        cog = result.get("cognition", {})
        pc3.metric("VLM结论", cog.get("conclusion","—") if cog and cog.get("ok") else "—")

        if result.get("heatmap_png") and os.path.exists(result["heatmap_png"]):
            st.image(result["heatmap_png"], caption="异常区域热力图（红色=异常位置）", width="stretch")

        # CLIP零样本缺陷类型分类
        cls = result.get("classification", {})
        if cls:
            st.markdown("##### 🤖 缺陷类型智能识别")
            import pandas as pd
            cls_df = pd.DataFrame([{"缺陷类型": k, "概率": f"{v*100:.1f}%"} for k, v in cls.items()])
            top_type = max(cls, key=cls.get)
            st.info(f"最可能：**{top_type}**（{cls[top_type]*100:.1f}%）")
            st.dataframe(cls_df, use_container_width=True, hide_index=True)

        if gi["route"] == "fast":
            st.info(f"异常分数 {result['score']:.4f} ≥ 阈值，直接输出定位（见热力图）。")
        elif cog and cog.get("ok"):
            vc = "verdict-fail" if "不合格" in cog.get("conclusion","") else "verdict-pass"
            st.markdown(f"""<div class="{vc}"><strong>📋 {cog.get('conclusion','')}</strong><br><br>{cog.get('defect_desc','')}</div>""",
                        unsafe_allow_html=True)
            st.markdown("**建议**：" + cog.get("suggestion",""))
            st.markdown("**引用**：" + " ".join(f'<span class="ref-tag">{r}</span>' for r in cog.get("ref_ids",[])))
            if cog.get("validation",{}).get("all_valid"):
                st.success("✅ 引用校验通过")
        else:
            st.warning(f"VLM失败：{cog.get('error','') if cog else '—'}")

        pdf_path = os.path.join(RESULTS_DIR, "reports", f"{os.path.splitext(sample_name or 'report')[0]}.pdf")
        report.generate_pdf(result, pdf_path, sample_name or "未知零件")
        with open(pdf_path, "rb") as f:
            st.download_button("📄 导出 PDF 报告", f, file_name=os.path.basename(pdf_path), mime="application/pdf")

        # 存到session供对话助手用
        st.session_state["_last_result"] = result
        st.session_state["_last_image"] = image_path
        st.session_state["_chat_history"] = []
        st.markdown('</div>', unsafe_allow_html=True)

# ============ 智能对话助手 ============
if st.session_state.get("_last_result"):
    st.markdown('<div class="card"><h3>💬 智能对话助手</h3>', unsafe_allow_html=True)

    r = st.session_state["_last_result"]
    cog = r.get("cognition", {})
    ctx = (f"结论：{cog.get('conclusion','快链路直出')}\n"
           f"缺陷描述：{cog.get('defect_desc','无')}\n"
           f"处置建议：{cog.get('suggestion','无')}\n"
           f"引用标准：{', '.join(cog.get('ref_ids',[])) or '无'}\n"
           f"异常分数：{r.get('score',0):.4f}")

    # 预设问题按钮（随机4个，可换一批）
    import random as _rnd
    st.caption("快捷提问（点击直接问，或在下方自由输入）：")
    _POOL = [
        "为什么判定这个结论？", "这个缺陷严重吗？",
        "GB/T 34528对这种缺陷怎么规定？", "需要返工还是报废？",
        "缺陷在图像的什么位置？", "这个异常分数是怎么算出来的？",
        "快链路和慢链路有什么区别？", "WinCLIP是怎么检测出来的？",
        "这个零件还能继续使用吗？", "打磨后能合格吗？",
        "引用的标准条款具体怎么说？", "同类型缺陷历史上怎么处理的？",
    ]
    if "_preset_idx" not in st.session_state:
        st.session_state["_preset_idx"] = _rnd.sample(range(len(_POOL)), 4)
    pcols = st.columns([1,1,1,1,0.6])
    for i in range(4):
        qtext = _POOL[st.session_state["_preset_idx"][i]]
        if pcols[i].button(qtext, key=f"preset_{i}"):
            st.session_state["_pending_q"] = qtext
    if pcols[4].button("🔀 换一批"):
        st.session_state["_preset_idx"] = _rnd.sample(range(len(_POOL)), 4)
        st.rerun()

    # 展示历史对话
    import re as _re
    def _expand_refs(text):
        """在回答下方自动展开引用的GB条款原文"""
        ids = _re.findall(r"GB34528-[A-Z0-9\-]+", text)
        if not ids:
            return
        with st.expander(f"📖 展开引用条款（{len(ids)}条）"):
            for rid in set(ids):
                hit = [x for x in (cog.get("retrieved") or []) if x.get("id") == rid]
                if hit:
                    h = hit[0]
                    st.markdown(f"**[{rid}]** P{h.get('page')} {h.get('section')}")
                    st.caption(h.get("text",""))
                else:
                    st.caption(f"[{rid}] （本次检索未命中该条款）")

    for idx, msg in enumerate(st.session_state.get("_chat_history", [])):
        role = "🧑 你" if msg["role"] == "user" else "🤖 助手"
        st.markdown(f"**{role}**：{msg['content']}")
        if msg["role"] == "assistant":
            _expand_refs(msg["content"])
            # 反馈按钮
            fb_key = f"fb_{idx}"
            fb = st.session_state.get("_feedback", {}).get(idx)
            fc1, fc2 = st.columns([0.1, 0.1])
            if fc1.button("👍", key=f"up_{idx}", disabled=(fb=="up")):
                st.session_state.setdefault("_feedback", {})[idx] = "up"
                report.save_feedback(st.session_state.get("_last_hist_path",""),
                                    {"message_idx": idx, "label": "up"})
                st.rerun()
            if fc2.button("👎", key=f"down_{idx}", disabled=(fb=="down")):
                st.session_state.setdefault("_feedback", {})[idx] = "down"
                cur_score = result.get("score", 0)
                report.save_feedback(st.session_state.get("_last_hist_path",""),
                                    {"message_idx": idx, "label": "down"})
                new_thr, changed = maybe_adjust_threshold("down", cur_score)
                if changed:
                    st.toast(f"已根据反馈自动调整阈值：{new_thr:.4f}", icon="🔧")
                st.rerun()
            if fb == "up":
                st.caption("✅ 已标记为有用")
            elif fb == "down":
                st.caption("❌ 已标记为需改进")

    # 输入框
    default_q = st.session_state.pop("_pending_q", "")
    extra_up = st.file_uploader("附加图片（可选，例如对比另一个零件）",
                                 type=["png","jpg","jpeg"], key="_chat_img")
    q = st.text_input("输入问题", value=default_q, key="_chat_input",
                      placeholder="例如：这张新图和刚才那张有什么区别？")
    extra_path = None
    if extra_up is not None:
        extra_path = os.path.join(RESULTS_DIR, "perception", "_chat_extra.png")
        with open(extra_path, "wb") as f:
            f.write(extra_up.getbuffer())
    if st.button("发送", type="primary", key="_chat_send") and q.strip():
        history = st.session_state.get("_chat_history", [])
        history.append({"role": "user", "content": q.strip()})
        st.session_state["_chat_history"] = history
        with st.spinner("助手思考中..."):
            stream_gen = vlm_api.chat_ask(
                st.session_state["_last_image"],
                [{"role": m["role"], "content": m["content"]} for m in history[:-1]],
                q.strip(), ctx, stream=True, extra_image=extra_path)
            answer = st.write_stream(stream_gen)
        history.append({"role": "assistant", "content": answer})
        st.session_state["_chat_history"] = history
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
