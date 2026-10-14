# -*- coding: utf-8 -*-
"""
对比实验2：纯VLM 直接判断（不经过WinCLIP）
==========================================
挑10张代表性图（5正常+5异常），每张直接调Qwen3.6-VL问"合格/不合格"，
统计准确率、平均耗时。
"""
import os, sys, json, time, base64
import requests
from PIL import Image

ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DASHSCOPE_MODEL

TEST_DIR = os.path.join(ROOT, "data", "mvtec", "metal_nut", "test")
OUT_DIR = os.path.join(ROOT, "results", "experiments")

# 挑10张代表性图
SAMPLES = [
    ("good", "000.png", 0),
    ("good", "005.png", 0),
    ("good", "010.png", 0),
    ("good", "015.png", 0),
    ("good", "020.png", 0),
    ("bent", "000.png", 1),
    ("color", "000.png", 1),
    ("flip", "000.png", 1),
    ("scratch", "000.png", 1),
    ("scratch", "010.png", 1),
]

PROMPT = """你是金属零件质检员。看这张金属螺母图片，判断它是否合格（表面无缺陷、形状完整）。
只输出JSON：{"verdict": "合格/不合格", "reason": "一句话理由"}
不要输出其他内容。"""

def ask_vlm(img_path):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": DASHSCOPE_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
        "temperature": 0.1,
        "enable_thinking": False,
    }
    t0 = time.time()
    r = requests.post(f"{DASHSCOPE_BASE_URL}/chat/completions",
                     headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                              "Content-Type": "application/json"},
                     json=payload, timeout=90)
    dt = time.time() - t0
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    return txt, dt

def main():
    records = []
    for cat, fn, label in SAMPLES:
        path = os.path.join(TEST_DIR, cat, fn)
        if not os.path.exists(path):
            print(f"跳过(不存在): {path}")
            continue
        try:
            txt, dt = ask_vlm(path)
        except Exception as e:
            print(f"失败 {cat}/{fn}: {e}")
            continue
        # 解析
        pred = 0 if "合格" in txt and "不合格" not in txt else 1
        # 更精确：找 verdict
        import re, json as _json
        m = re.search(r'\{[\s\S]*\}', txt)
        if m:
            try:
                j = _json.loads(m.group())
                v = j.get("verdict", "")
                pred = 0 if "合格" in v and "不合格" not in v else 1
            except Exception:
                pass
        correct = (pred == label)
        rec = {"category": cat, "file": fn, "label": label,
               "pred": pred, "correct": correct, "time_s": round(dt,2),
               "raw": txt[:200]}
        records.append(rec)
        print(f"{cat}/{fn}  真值={'正常' if label==0 else '异常'}  "
              f"VLM={'正常' if pred==0 else '异常'}  {'✓' if correct else '✗'}  {dt:.1f}s")

    correct = sum(1 for r in records if r["correct"])
    acc = correct / len(records) if records else 0
    avg_t = sum(r["time_s"] for r in records) / len(records) if records else 0
    summary = {
        "n": len(records),
        "accuracy": round(acc, 4),
        "avg_time_s": round(avg_t, 2),
        "records": records,
    }
    with open(os.path.join(OUT_DIR, "exp2_vlm_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n纯VLM准确率: {acc*100:.1f}% ({correct}/{len(records)})")
    print(f"平均耗时: {avg_t:.1f}s/张")

if __name__ == "__main__":
    main()
