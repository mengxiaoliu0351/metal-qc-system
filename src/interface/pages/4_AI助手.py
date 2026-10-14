# -*- coding: utf-8 -*-
"""系统级AI助手：ChatGPT/DeepSeek样式聊天界面"""
import os, base64, json, random as _rnd
from datetime import datetime
import streamlit as st
import requests
from interface.common import CSS, HERO_HTML, sidebar, THRESHOLD
from interface import report
from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DASHSCOPE_MODEL

st.set_page_config(page_title="AI助手", page_icon="🤖", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)
sidebar()

st.markdown("## 🤖 智能质检助手")

# ============ 系统上下文 ============
def _retrieve_relevant(question: str, k: int = 5):
    """轻量关键词检索：从历史里挑和问题最相关的k条。
    不依赖向量模型，按关键词命中数打分。"""
    hist = report.list_history(limit=200)
    q_words = [w for w in ["合格","不合格","缺陷","划痕","弯曲","色差","裂纹",
                           "inclusion","crazing","patches","scratches",
                           "快链路","慢链路","阈值","反馈","今日","昨天",
                           "GB","标准","人工","复核","分数"] if w in question]
    if not q_words:
        return hist[:5]  # 没关键词，返回最近5条
    scored = []
    for h in hist:
        text = " ".join([
            str(h.get("sample_name","")),
            str((h.get("cognition") or {}).get("conclusion","")),
            str((h.get("cognition") or {}).get("defect_desc","")),
            str((h.get("cognition") or {}).get("suggestion","")),
        ])
        hit = sum(1 for w in q_words if w in text)
        scored.append((hit, h))
    scored.sort(key=lambda x: -x[0])
    return [h for hit, h in scored[:k] if hit > 0] or hist[:5]


def _build_system_context(question: str = ""):
    hist = report.list_history(limit=50)
    total = len(hist)
    today = datetime.now().strftime("%Y-%m-%d")
    today_recs = [h for h in hist if h.get("timestamp","").startswith(today)]
    fail = sum(1 for h in hist if (h.get("cognition") or {}).get("conclusion","") == "不合格")
    fast = sum(1 for h in hist if h.get("gate",{}).get("route") == "fast")
    lines = [
        "【系统配置】",
        f"- 判定阈值：{THRESHOLD}",
        f"- 检测方式：智能图像识别 + AI缺陷分析",
        f"- 分级策略：高置信快速出结果，低置信深度分析",
        "",
        f"【历史统计（最近{total}条）】",
        f"- 今日检测：{len(today_recs)}条",
        f"- 快速/深度：{fast}/{total-fast}",
        f"- 不合格：{fail}条",
        "",
    ]
    if question:
        relevant = _retrieve_relevant(question, k=5)
        lines.append(f"【和问题「{question}」最相关的历史】")
        for h in relevant:
            ts = h.get("timestamp","")[:16].replace("T"," ")
            cog = h.get("cognition") or {}
            lines.append(f"- {ts} | {h.get('sample_name','')} | 分数{h.get('score',0):.4f} | "
                         f"{h.get('gate',{}).get('route','?')} | {cog.get('conclusion','快链路')}"
                         f" | {cog.get('defect_desc','')[:50]}")
    else:
        lines.append("【最近检测明细】")
        for h in hist[:10]:
            ts = h.get("timestamp","")[:16].replace("T"," ")
            lines.append(f"- {ts} | {h.get('sample_name','')} | 分数{h.get('score',0):.4f} | "
                         f"{h.get('gate',{}).get('route','?')} | {(h.get('cognition') or {}).get('conclusion','快链路')}")
    return "\n".join(lines)

# ============ 初始化 ============
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是智能质检助手。可以问我：\n- 今天检测了多少个？合格率多少？\n- 什么样算不合格？\n- 上传零件图让我看看"}
    ]

# ============ 渲染历史（聊天气泡）============
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============ 快捷问题 ============
_POOL = [
    "今天检测了多少个？合格率多少？",
    "什么样算不合格？判定规则是什么？",
    "快链路和慢链路有什么区别？",
    "GB/T 34528-2017主要管什么？",
]
cols = st.columns(4)
for i, q in enumerate(_POOL):
    if cols[i].button(q, key=f"quick_{i}"):
        st.session_state._pending_q = q

# ============ 可选图片上传 ============
up = st.file_uploader("📎 上传零件图（可选）", type=["png","jpg","jpeg"], key="chat_img")
extra_path = None
if up is not None:
    extra_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                              "results", "perception", "_chat_upload.png")
    os.makedirs(os.path.dirname(extra_path), exist_ok=True)
    with open(extra_path, "wb") as f:
        f.write(up.getbuffer())
    st.image(extra_path, caption="已上传", width=200)

# ============ 输入框（底部）============
default_q = st.session_state.pop("_pending_q", "")
prompt = st.chat_input(default_q or "输入你的问题...")

def _stream_ask(history, question, image_path=None):
    sys_p = ("你是双链路智能质检系统的管理助手。基于以下系统配置和历史记录回答。\n"
             "要求：中文、简洁专业、3-5句话；统计数字必须基于下方历史数据不要编造。\n\n"
             f"{_build_system_context(question)}")
    msgs = [{"role": "system", "content": sys_p}]
    for m in history:
        msgs.append(m)
    content = []
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": question})
    msgs.append({"role": "user", "content": content if content else question})
    payload = {"model": DASHSCOPE_MODEL, "messages": msgs,
               "temperature": 0.3, "enable_thinking": False, "stream": True}
    resp = requests.post(f"{DASHSCOPE_BASE_URL}/chat/completions",
                         headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                                  "Content-Type": "application/json"},
                         json=payload, timeout=90, stream=True)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            d = json.loads(data)["choices"][0]["delta"].get("content", "")
            if d:
                yield d
        except Exception:
            continue

# ============ 处理提问 ============
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        answer = st.write_stream(_stream_ask(history, prompt, extra_path))
    st.session_state.messages.append({"role": "assistant", "content": answer})

# 清空按钮
if st.button("🗑️ 新对话"):
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是智能质检助手。有什么可以帮你的？"}
    ]
    st.rerun()
