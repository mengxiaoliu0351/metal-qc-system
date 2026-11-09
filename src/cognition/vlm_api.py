# -*- coding: utf-8 -*-
"""
VLM 统一调用模块：豆包（火山方舟）/ 通义（DashScope）/ 本地Ollama
=================================================================
三个后端都封装成 analyze(image_path, prompt) -> dict 的同一接口，
切换只改 config.VLM_PROVIDER，上层代码不用动。

返回结构：
    {
      "ok": True,
      "conclusion": "不合格",
      "defect_desc": "零件中心区域可见一条线状划痕...",
      "ref_ids": ["QC-NOTE-04"],
      "suggestion": "建议打磨清除后复测深度...",
      "raw": "VLM原始输出文本",
    }
API 失败时 ok=False 并带 error 信息，由上层决定走缓存/降级。
"""
import base64
import io
import json
import os
import re

import requests

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DOUBAO_API_KEY, DOUBAO_BASE_URL, DOUBAO_MODEL,
                    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DASHSCOPE_MODEL,
                    OLLAMA_URL, OLLAMA_MODEL)

# 强制 VLM 输出结构化 JSON 的 Prompt
SYSTEM_PROMPT = """你是金属表面缺陷质检分析助手。根据用户提供的缺陷区域图像和检索到的知识材料（来源类型会注明），输出质检结论。

必须严格输出以下 JSON 格式（不要输出其他内容）：
{
  "conclusion": "合格/不合格/需复核",
  "defect_type": "划痕/弯曲/色差/裂纹/凸起/凹陷/正常/其他",
  "defect_desc": "对缺陷形态、位置、程度的自然语言描述",
  "ref_ids": ["引用ID1", "引用ID2"],
  "suggestion": "处置建议"
}

要求：
1. ref_ids 只能从本次提供的知识材料 ID 中选取，禁止编造不存在的 ID；
2. defect_desc 基于图像实际内容描述，不超过80字；
3. defect_type 必须从给定选项中选一个，不准自造；
4. suggestion 结合材料给出观察与核验建议。项目知识卡是教学示例，标准目录只有编号名称，都不是国标原文，不得据此引用条款或编造限值。
5. 单张无标尺图像不能量化实际尺寸或深度，不能给出毫米、照度等未经证实的限值。"""

VISUAL_SYSTEM_PROMPT = """你是金属零件外观观察助手。先根据原图判断可见外观，再说明需要核验的事项。
第一张图是整件原图，后续图片是原图局部原色裁剪。局部由异常响应选取，仅提示观察位置。
正常零件也有高响应位置，反光、加工纹理、螺纹及网格本身均可产生响应。保持独立观察。
结合整图核实局部，区分连续规则纹理、反光与真实的非规则划伤、裂口、变形或污染。
知识卡用于描述和后续核验。缺少标准原文、实测尺寸时，将 standards_assessment 设为需核验，
仍应独立回答图片上是否有明确外观异常。不要仅因没有标准条款就将清楚的外观观察判为不确定。
看不清、相互矛盾或无法区分正常结构和缺陷时，visual_assessment 必须为 uncertain。
无异常结论需要同时检查完整工件，不能仅看两个裁剪。未显示的背面和内部不在判断范围内。
严格返回一个JSON对象：
{
  "visual_assessment": "visible_defect/no_visible_defect/uncertain",
  "image_adequacy": "adequate/inadequate",
  "conclusion": "不合格/合格/需复核",
  "defect_type": "划痕/弯曲/色差/裂纹/凸起/凹陷/正常/其他",
  "defect_location": "描述原图中的位置，无法定位时写不确定",
  "defect_desc": "80字内的具体可见证据，区分结构或反光与异常",
  "alternative_explanation": "可能的正常纹理、反光等解释，不能排除时明确说明",
  "standards_assessment": "需核验",
  "ref_ids": [],
  "suggestion": "后续核验或处置建议"
}
conclusion 是本图可见外观的辅助判断。visible_defect 对应不合格，no_visible_defect 对应合格，
uncertain 或 inadequate 对应需复核。no_visible_defect 时 defect_type 为正常。
它不代表整件工业质量认证。具体归档由独立作业规则决定。
ref_ids 只能选用本次材料的真实编号，无相关材料时留空。教学知识卡和标准目录不是国标原文。
单张无标尺图像不能量化实际尺寸、深度、公差，不编造标准号、条款、页码或合格限值。"""


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_openai_compat(base_url, api_key, model, image_path, prompt, timeout=120,
                        focus_images=None, system_prompt=SYSTEM_PROMPT, request_metadata=None):
    """OpenAI 兼容接口（豆包/通义都支持）"""
    from PIL import Image
    with Image.open(image_path) as original:
        images = [original.convert("RGB"), *(focus_images or [])]
    content = []
    for index, image in enumerate(images):
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        content.extend([
            {"type": "text", "text": "整件原图" if index == 0 else f"原色局部{index}"},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")}},
        ])
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "enable_thinking": False,  # 关掉思考模式，演示要快
    }
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if request_metadata is not None:
        request_metadata.update(model=data.get("model", model), usage=data.get("usage", {}))
    return data["choices"][0]["message"]["content"]


def _call_ollama(image_path, prompt, timeout=120):
    """本地 Ollama（断网兜底）"""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt,
             "images": [_encode_image(image_path)]},
        ],
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _parse_json(text: str) -> dict:
    """从 VLM 输出中稳健提取 JSON（容忍代码块/多余文字）"""
    text = text.strip()
    # 去掉 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1)
    # 找第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def analyze(image_path, prompt, backend=None, provider=None, *, focus_images=None,
            visual_review=False) -> dict:
    """统一入口：分析缺陷裁剪图，返回结构化结论"""
    from config import VLM_PROVIDER as _CFG_PROVIDER, VLM_BACKEND as _CFG_BACKEND
    backend = backend or _CFG_BACKEND
    provider = provider or _CFG_PROVIDER
    metadata = {}

    try:
        if backend == "local":
            raw = _call_ollama(image_path, prompt)
        elif provider == "dashscope":
            if not DASHSCOPE_API_KEY or "PASTE" in DASHSCOPE_API_KEY:
                return {"ok": False, "error": "未配置通义 API Key，请在 config.py 或环境变量设置"}
            raw = _call_openai_compat(DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY,
                                      DASHSCOPE_MODEL, image_path, prompt,
                                      focus_images=focus_images,
                                      system_prompt=VISUAL_SYSTEM_PROMPT if visual_review else SYSTEM_PROMPT,
                                      request_metadata=metadata)
        else:
            if not DOUBAO_API_KEY or "PASTE" in DOUBAO_API_KEY:
                return {"ok": False, "error": "未配置豆包 API Key，请在 config.py 或环境变量设置"}
            raw = _call_openai_compat(DOUBAO_BASE_URL, DOUBAO_API_KEY,
                                      DOUBAO_MODEL, image_path, prompt,
                                      focus_images=focus_images,
                                      system_prompt=VISUAL_SYSTEM_PROMPT if visual_review else SYSTEM_PROMPT,
                                      request_metadata=metadata)

        parsed = _parse_json(raw)
        if not parsed:
            return {"ok": False, "error": "VLM 未返回结构化 JSON", "raw": raw}

        result = {
            "ok": True,
            "conclusion": parsed.get("conclusion", "需复核"),
            "defect_type": parsed.get("defect_type", "其他"),
            "defect_desc": parsed.get("defect_desc", ""),
            "ref_ids": parsed.get("ref_ids", []),
            "suggestion": parsed.get("suggestion", ""),
            "raw": raw,
            "model": metadata.get("model", OLLAMA_MODEL if backend == "local" else ""),
            "usage": metadata.get("usage", {}),
        }
        if visual_review:
            result.update({key: parsed.get(key) for key in (
                "visual_assessment", "image_adequacy", "defect_location", "alternative_explanation")})
            result.update(review_version="focus-v2", standards_assessment="需核验")
            allowed_types = {"划痕", "弯曲", "色差", "裂纹", "凸起", "凹陷", "正常", "其他"}
            if not isinstance(result["defect_type"], str) or result["defect_type"] not in allowed_types:
                result["reported_defect_type"] = result["defect_type"]
                result["defect_type"] = "其他"
                result["visual_assessment"] = "uncertain"
            suggestion = str(result.get("suggestion") or "")
            if "条款" in suggestion or re.search(r"第\s*[0-9一二三四五六七八九十]+[.0-9]*\s*条", suggestion):
                result["unverified_standard_claim"] = True
                result["suggestion"] = "请人工核对原图中的外观证据，并结合实物测量和适用产品规范复检。"
            expected = {"visible_defect": "不合格", "no_visible_defect": "合格", "uncertain": "需复核"}
            adequate = result["image_adequacy"] == "adequate"
            consistent = expected.get(result["visual_assessment"]) == result["conclusion"]
            normal_consistent = result["visual_assessment"] != "no_visible_defect" or result["defect_type"] == "正常"
            evidence = isinstance(result["defect_desc"], str) and bool(result["defect_desc"].strip())
            if not (adequate and consistent and normal_consistent and evidence):
                result["reported_conclusion"] = result["conclusion"]
                result["conclusion"] = "需复核"
                result["visual_assessment"] = "uncertain"
        return result
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "API 请求超时（120秒），请检查网络"}
    except Exception as e:
        from web.public_demo import PUBLIC_REQUEST
        return {"ok": False, "error": "AI复核暂不可用，请人工复核。" if PUBLIC_REQUEST.get() else f"VLM 调用失败: {e}"}


def build_prompt(defect_hint: str, retrieved_texts: list) -> str:
    """构造认知层 Prompt：缺陷提示 + 检索到的标准片段"""
    prompt = f"请分析以下金属表面缺陷区域。\n缺陷定位提示：{defect_hint}\n\n"
    prompt += "可引用的知识材料（教学知识卡与标准目录均不是国标原文）：\n"
    for t in retrieved_texts:
        prompt += f"{t}\n"
    prompt += "\n请基于图像与上述材料输出辅助判断，证据不足时结论为需复核（JSON格式）。"
    return prompt


def chat_ask(image_path: str, history: list, user_question: str, context_summary: str = "",
             stream: bool = False, extra_image: str = None):
    """多轮对话：传入原图、历史消息列表、用户新问题，返回VLM回答文本（或生成器）。
    history 格式：[{"role":"user"/"assistant","content":"..."}, ...]
    context_summary：当前检测结果摘要（结论/缺陷/引用），作为系统上下文。
    stream=True 时返回逐字生成器。
    extra_image：本轮额外上传的图片路径（可选），会和问题一起发给VLM。
    """
    sys_prompt = ("你是金属表面缺陷质检智能助手。基于提供的零件图像和以下检测结果，"
                  "用中文简洁回答用户追问。不要输出JSON，自然语言即可。\n\n"
                  f"【当前检测结果】\n{context_summary}\n\n"
                  "回答要求：专业、简洁（3-5句话），只能引用当前上下文提供的真实 ID。不得编造标准号、条款、页码或尺寸限值。")

    msgs = [{"role": "system", "content": sys_prompt}]
    for m in history:
        msgs.append(m)

    # 最新user消息：带原图 + 可能的附加图
    user_content = []
    user_content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image_path)}"}})
    if extra_image:
        user_content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(extra_image)}"}})
    user_content.append({"type": "text", "text": user_question})
    msgs.append({"role": "user", "content": user_content})

    payload = {
        "model": DASHSCOPE_MODEL,
        "messages": msgs,
        "temperature": 0.3,
        "enable_thinking": False,
        "stream": stream,
    }
    try:
        resp = requests.post(
            f"{DASHSCOPE_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                     "Content-Type": "application/json"},
            json=payload, timeout=90, stream=stream)
        resp.raise_for_status()
        if stream:
            def gen():
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
                        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
            return gen()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        if stream:
            def gen_err():
                yield f"（对话调用失败：{e}）"
            return gen_err()
        return f"（对话调用失败：{e}）"


def quick_describe(image_path: str) -> str:
    """快链路用：VLM用5-10个字描述缺陷，不要求JSON。1-2秒返回。"""
    try:
        payload = {
            "model": DASHSCOPE_MODEL,
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image_path)}"}},
                    {"type": "text",
                     "text": "这是金属零件表面。用5-10个字描述可见的异常（划痕/弯曲/色差/裂纹/凸起/正常）。只回答描述，不要其他。"}
                ]}
            ],
            "temperature": 0.1,
            "enable_thinking": False,
            "max_tokens": 30,
        }
        resp = requests.post(f"{DASHSCOPE_BASE_URL}/chat/completions",
                             headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                                      "Content-Type": "application/json"},
                             json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(描述暂不可用: {e})"
    # 自测：无图时给出提示
    print("VLM API 模块已加载")
    print("后端:", "local(Ollama)" if os.environ.get("VLM_BACKEND") == "local" else "api")
