# -*- coding: utf-8 -*-
"""
检测历史记录 + PDF 报告导出
=========================
- 历史：每次检测存 results/history/YYYY-MM-DD/HHMMSS_<名称>.json
- PDF：reportlab 生成正式质检报告（图+热力图+结论+引用+建议）
"""
import json
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 注册中文字体（Windows 自带微软雅黑）
_FONT_NAME = "Helvetica"
for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]:
    if os.path.exists(fp):
        try:
            pdfmetrics.registerFont(TTFont("CN", fp))
            _FONT_NAME = "CN"
            break
        except Exception:
            continue

HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "..", "results", "history")
HISTORY_DIR = os.path.abspath(HISTORY_DIR)


def save_history(result: dict, sample_name: str = ""):
    """存一条检测历史，返回文件路径"""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    day = datetime.now().strftime("%Y-%m-%d")
    day_dir = os.path.join(HISTORY_DIR, day)
    os.makedirs(day_dir, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    name = sample_name or os.path.basename(result.get("input_png", "image"))
    name = os.path.splitext(name)[0]
    path = os.path.join(day_dir, f"{ts}_{name}.json")
    record = {
        "timestamp": datetime.now().isoformat(),
        "sample_name": name,
        **result,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    return path


def save_feedback(history_path: str, feedback: dict):
    """把用户反馈追加到历史记录，并触发反哺逻辑。
    feedback: {"message_idx": int, "label": "up"/"down", "comment": str}
    反哺：
    - 👎 自动给该记录打 needs_review 标记（需要人工复核）
    - 👎 把原图复制到困难样本库（results/difficult_samples/），供后续模型迭代
    - 👎 累计达到阈值时，返回 "建议重建记忆库" 信号
    """
    if not os.path.exists(history_path):
        return {"ok": False, "reason": "history not found"}
    with open(history_path, encoding="utf-8") as f:
        rec = json.load(f)
    rec.setdefault("feedback", []).append({
        **feedback,
        "ts": datetime.now().isoformat(),
    })
    # 反哺：👎 标记需要人工复核 + 存困难样本
    if feedback.get("label") == "down":
        rec["needs_review"] = True
        # 把原图复制到困难样本库
        try:
            src = rec.get("input_png")
            if src and os.path.exists(src):
                diff_dir = os.path.join(os.path.dirname(HISTORY_DIR), "difficult_samples")
                os.makedirs(diff_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                dst = os.path.join(diff_dir, f"{ts}_{rec.get('sample_name','img')}.png")
                from shutil import copyfile
                copyfile(src, dst)
                rec["difficult_sample_path"] = dst
        except Exception:
            pass
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2, default=str)
    return {"ok": True, "needs_review": rec.get("needs_review", False)}


def get_feedback_summary():
    """扫描全部历史，返回反馈统计和反哺建议。
    返回：
    - total_up/total_down: 👍👎总数
    - needs_review: 被标记需复核的记录数
    - suggestion: 系统级建议（如"某产品建议重建库"）
    """
    total_up = total_down = needs_review = 0
    down_by_score_range = {"低分区(<0.2)": 0, "阈值附近(0.2-0.3)": 0, "高分区(>0.3)": 0}
    records = list_history(limit=10000)
    for r in records:
        for fb in r.get("feedback", []):
            if fb.get("label") == "up":
                total_up += 1
            elif fb.get("label") == "down":
                total_down += 1
                s = float(r.get("score", 0))
                if s < 0.2:
                    down_by_score_range["低分区(<0.2)"] += 1
                elif s < 0.3:
                    down_by_score_range["阈值附近(0.2-0.3)"] += 1
                else:
                    down_by_score_range["高分区(>0.3)"] += 1
        if r.get("needs_review"):
            needs_review += 1
    suggestion = ""
    if down_by_score_range["低分区(<0.2)"] >= 3:
        suggestion = "低分区(分数<0.2)被多次判👎：建议重新检查记忆库，可能有缺陷件被误判为正常"
    elif down_by_score_range["高分区(>0.3)"] >= 3:
        suggestion = "高分区(分数>0.3)被多次判👎：建议检查VLM慢链路描述是否准确"
    elif down_by_score_range["阈值附近(0.2-0.3)"] >= 5:
        suggestion = "阈值附近大量👎：建议重新校准门控阈值"
    return {
        "total_up": total_up, "total_down": total_down,
        "needs_review": needs_review,
        "down_by_score_range": down_by_score_range,
        "suggestion": suggestion,
    }


def list_history(limit: int = 20):
    """列出最近的检测记录，按时间倒序"""
    records = []
    if not os.path.isdir(HISTORY_DIR):
        return records
    for day in sorted(os.listdir(HISTORY_DIR), reverse=True):
        day_dir = os.path.join(HISTORY_DIR, day)
        if not os.path.isdir(day_dir):
            continue
        for fn in sorted(os.listdir(day_dir), reverse=True):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(day_dir, fn), encoding="utf-8") as f:
                        rec = json.load(f)
                    rec["_file"] = os.path.splitext(fn)[0]  # 用于详情回看
                    rec["_day"] = day
                    records.append(rec)
                except Exception:
                    pass
            if len(records) >= limit:
                return records
    return records


def label_active_learning(record: dict, label: str):
    """主动学习：给一条记录打标注（合格/不合格），并写回JSON"""
    ts = record.get("timestamp", "")
    if not ts:
        return False
    # 找文件路径
    for day in sorted(os.listdir(HISTORY_DIR), reverse=True):
        day_dir = os.path.join(HISTORY_DIR, day)
        if not os.path.isdir(day_dir):
            continue
        for fn in os.listdir(day_dir):
            if fn.endswith(".json"):
                p = os.path.join(day_dir, fn)
                try:
                    with open(p, encoding="utf-8") as f:
                        rec = json.load(f)
                    if rec.get("timestamp") == ts:
                        rec["active_learning_labeled"] = True
                        rec["active_learning_label"] = label
                        rec["active_learning_ts"] = datetime.now().isoformat()
                        with open(p, "w", encoding="utf-8") as f:
                            json.dump(rec, f, ensure_ascii=False, indent=2, default=str)
                        return True
                except Exception:
                    continue
    return False


def generate_pdf(result: dict, output_path: str, sample_name: str = "未知零件"):
    """生成正式质检报告 PDF"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                fontName=_FONT_NAME, fontSize=18, spaceAfter=12)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"],
                         fontName=_FONT_NAME, fontSize=13, spaceAfter=6,
                         textColor=colors.HexColor("#2c5364"))
    body = ParagraphStyle("body", parent=styles["Normal"],
                          fontName=_FONT_NAME, fontSize=10.5, leading=16)
    small = ParagraphStyle("small", parent=styles["Normal"],
                           fontName=_FONT_NAME, fontSize=9, textColor=colors.grey)

    story = []
    story.append(Paragraph("金属表面缺陷智能质检报告", title_style))
    story.append(Paragraph(f"样品名称：{sample_name} &nbsp;&nbsp;|&nbsp;&nbsp; "
                           f"检测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small))
    story.append(Spacer(1, 0.5*cm))

    # 基本信息表
    score = result.get("score", 0.0)
    gate = result.get("gate", {})
    cog = result.get("cognition", {})
    info = [
        ["异常分数", f"{score:.4f}", "门控阈值", f"{gate.get('threshold', 0.2243):.4f}"],
        ["路由路径", "快链路（直接输出）" if gate.get("route") == "fast" else "慢链路（深度分析）",
         "路由标签", gate.get("label", "")],
        ["VLM结论", cog.get("conclusion", "—") if cog else "—",
         "引用校验", "通过" if cog and cog.get("validation", {}).get("all_valid") else "—"],
    ]
    t = Table(info, colWidths=[3*cm, 5*cm, 3*cm, 5*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), _FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f5")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#eef2f5")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # 图像
    if result.get("input_png") and os.path.exists(result["input_png"]):
        story.append(Paragraph("① 待检图像", h2))
        story.append(Image(result["input_png"], width=10*cm, height=10*cm))
        story.append(Spacer(1, 0.3*cm))
    if result.get("heatmap_png") and os.path.exists(result["heatmap_png"]):
        story.append(Paragraph("② WinCLIP 异常热力图", h2))
        story.append(Image(result["heatmap_png"], width=10*cm, height=10*cm))
        story.append(Spacer(1, 0.3*cm))

    # VLM 分析
    if cog and cog.get("ok"):
        story.append(Paragraph("③ VLM 缺陷分析", h2))
        story.append(Paragraph(f"<b>判定：</b>{cog.get('conclusion', '')}", body))
        story.append(Paragraph(f"<b>缺陷描述：</b>{cog.get('defect_desc', '')}", body))
        story.append(Paragraph(f"<b>处置建议：</b>{cog.get('suggestion', '')}", body))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(f"<b>引用标准：</b>{', '.join(cog.get('ref_ids', []))}", body))
        if cog.get("retrieved"):
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph("④ 检索到的标准条款", h2))
            for r in cog["retrieved"]:
                story.append(Paragraph(
                    f"[{r['id']}] P{r['page']} {r['section']}（相似度 {r.get('score', 0):.3f}）："
                    f"{r['text'][:120]}...", body))
    elif gate.get("route") == "fast":
        story.append(Paragraph("③ 快链路输出", h2))
        story.append(Paragraph(
            f"异常分数 {score:.4f} ≥ 阈值 {gate.get('threshold', 0.2243):.4f}，"
            f"直接输出缺陷定位（见热力图）。人工复核后签发。", body))

    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("—— 本报告由双链路智能质检辅助系统生成，最终结论由人工确认签发 ——", small))

    doc.build(story)
    return output_path
