# -*- coding: utf-8 -*-
"""
检测历史记录 + PDF 报告导出
=========================
- 历史：每次检测存 results/history/YYYY-MM-DD/HHMMSS_<名称>.json
- PDF：reportlab 生成正式质检报告（图+热力图+结论+引用+建议）
"""
import json
import os
import re
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
for fp in [os.environ.get("PDF_CJK_FONT", ""),
           r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc",
           "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
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
    name = (sample_name or os.path.basename(result.get("input_png", "image"))).replace("\\", "/").split("/")[-1]
    name = os.path.splitext(name)[0]
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)[:100] or 'image'
    import uuid
    suffix = result.get("session_id") or uuid.uuid4().hex[:10]
    path = os.path.join(day_dir, f"{ts}_{name}_{suffix}.json")
    record = {
        "timestamp": datetime.now().isoformat(),
        "sample_name": name,
        **result,
    }
    from review.workflow import write
    write(path, record)
    return path


def save_feedback(history_path: str, feedback: dict):
    """保存反馈意见；负反馈加入待复核与困难样本，不修改标签或模型。"""
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
        rec["review_reopened"] = True
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
    from review.workflow import write
    write(history_path, rec)
    return {"ok": True, "needs_review": rec.get("needs_review", False)}


def get_feedback_summary():
    """扫描全部历史，返回反馈统计和反哺建议。
    返回：
    - total_up/total_down: 👍👎总数
    - needs_review: 被标记需复核的记录数
    - suggestion: 系统级建议（如"某产品建议重建库"）
    """
    total_up = total_down = needs_review = 0
    down_by_score_range = {"低风险": 0, "灰区": 0, "高风险": 0}
    records = list_history(limit=10000)
    for r in records:
        for fb in r.get("feedback", []):
            if fb.get("label") == "up":
                total_up += 1
            elif fb.get("label") == "down":
                total_down += 1
                route = r.get("gate", {}).get("route", "")
                key = {"low_risk": "低风险", "vlm_review": "灰区", "high_risk": "高风险"}.get(route, "灰区")
                down_by_score_range[key] += 1
        if r.get("needs_review") and not (r.get("human_verdict") or {}).get("confirmed"):
            needs_review += 1
    suggestion = "请在人工复核中确认真实标签；批准用于迭代后再评估候选模型，反馈不会直接改变模型。" if total_down else ""
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
    """Export saved AI evidence and the separately recorded human final label."""
    from xml.sax.saxutils import escape
    from review.workflow import original_prediction
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=1.8*cm,
                            rightMargin=1.8*cm, topMargin=1.6*cm, bottomMargin=1.7*cm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("QCtitle", parent=styles["Title"], fontName=_FONT_NAME,
                           fontSize=17, leading=24, spaceAfter=10, textColor=colors.HexColor("#233b4a"))
    heading = ParagraphStyle("QCheading", parent=styles["Heading2"], fontName=_FONT_NAME,
                             fontSize=11, leading=17, spaceBefore=12, spaceAfter=7,
                             textColor=colors.HexColor("#355b70"))
    body = ParagraphStyle("QCbody", parent=styles["Normal"], fontName=_FONT_NAME,
                          fontSize=9.5, leading=15, spaceAfter=5)
    small = ParagraphStyle("QCsmall", parent=body, fontSize=8, leading=12,
                           textColor=colors.HexColor("#617582"))
    def text(value):
        return escape(str(value if value is not None else "—")).replace("\n", "<br/>")
    def para(value, style=body):
        return Paragraph(text(value), style)
    def number(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) else "—"
    def styled_table(rows, widths):
        table = Table([[para(cell, small) for cell in row] for row in rows], colWidths=widths)
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#cbd6dd")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#edf2f5")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        return table
    gate = result.get("gate") or {}
    cog = result.get("cognition") or {}
    human = result.get("human_verdict") or {}
    from review.disposition import effective_disposition
    disposition = effective_disposition(result)
    confirmed = human.get("confirmed") and human.get("verdict") in ("ok", "defect") and not disposition["requires_review"]
    route = gate.get("route")
    route_name = {"low_risk": "低风险 / 本地提示", "vlm_review": "灰区 / VLM辅助复核",
                  "high_risk": "高风险 / 本地提示", "fast": "旧版本地链路",
                  "slow": "旧版深度链路", "general": "通用视觉直检", "input_review": "输入待核验", "processing_error": "处理异常"}.get(route, "未记录")
    prediction = original_prediction(result)
    system_label = {"normal": "正常", "defect": "缺陷"}.get(prediction, "未记录")
    story = [Paragraph("金属表面缺陷检验报告", title),
             para(disposition["label"], small),
             Spacer(1, .35*cm)]
    story.append(styled_table([
        ["样品", sample_name, "产品", result.get("product_name", "未记录")],
        ["原检测时间", result.get("timestamp", "未记录"), "模型版本", result.get("perception_version", "未记录 / 历史版本")],
        ["异常评分", number(result.get("score")), "系统原始判定", system_label],
        ["下 / 上阈值", number(gate.get("threshold")) + " / " + number(gate.get("t_high")), "风险分流", route_name],
        ["AI辅助建议", cog.get("conclusion", "未记录"), "是否调用VLM", "是" if result.get('vlm_called', route in ("vlm_review", "slow", "general")) else "否"]
    ], [2.5*cm, 5.9*cm, 2.5*cm, 6.5*cm]))
    story.append(Paragraph("一、作业处置与人工核验", heading))
    story.append(para("处理状态：" + disposition["label"] + "。" + str(disposition.get("reason", ""))))
    if result.get("disposition"):
        story.append(para("作业模式：" + ("验证全检" if disposition.get("mode") == "validation" else "辅助作业") +
                          "；抽查比例：" + str(round(disposition.get("sample_rate", 0) * 100)) + "%" +
                          "；本件" + ("被抽中核验" if disposition.get("sampled") else "未被随机抽中") + "。", small))
    if confirmed:
        human_label = "正常 / 合格" if human["verdict"] == "ok" else "存在缺陷 / 不合格"
        agreement = human.get("agreement")
        if agreement is None and prediction:
            agreement = prediction == ("normal" if human["verdict"] == "ok" else "defect")
        story.append(styled_table([
            ["最终真实标签", human_label, "与系统判定", "一致" if agreement is True else "不一致" if agreement is False else "无法比较"],
            ["复核人员", human.get("reviewer", "未署名"), "复核时间", human.get("ts", "未记录")],
            ["缺陷类型", human.get("defect_type") or "—", "迭代授权", "批准用于离线迭代评估" if human.get("use_for_iteration") else "仅保存复核标签"]
        ], [2.5*cm, 5.9*cm, 2.5*cm, 6.5*cm]))
        if human.get("note"):
            story.append(Spacer(1, .2*cm)); story.append(para("核验依据：" + human["note"]))
        story.append(para("该记录表示人工标签确认，不等同于数字签名或模型已更新；候选版本需另行评估和应用。", small))
    elif disposition["status"] in ("auto_normal", "auto_isolated"):
        story.append(para("本次系统判断已自动归档或隔离，尚无人工真实标签。自动结果不计作人工核验正确率，不进入反馈训练标签池。"))
    else:
        story.append(para("尚未记录人工最终标签。本报告中的系统分数与AI建议不能替代检验人员的最终判断。"))
    pictures = []
    for field, label in (("input_png", "工件原图"), ("heatmap_png", "局部异常热力图")):
        image_path = result.get(field)
        if image_path and os.path.isfile(image_path):
            from PIL import Image as PILImage
            with PILImage.open(image_path) as im:
                w, h = im.size
            scale = min(7.2*cm/w, 6.7*cm/h)
            pictures.append([para(label, small), Image(image_path, width=w*scale, height=h*scale)])
    if pictures:
        story.append(Paragraph("二、检验图像证据", heading))
        layout = Table([[picture[0] for picture in pictures], [picture[1] for picture in pictures]],
                       colWidths=[17.4*cm/len(pictures)]*len(pictures))
        layout.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"),
                                   ("VALIGN", (0,0), (-1,-1), "TOP"),
                                   ("BOTTOMPADDING", (0,0), (-1,-1), 8)]))
        story.append(layout)
        story.append(para("热力图仅表示相对异常响应，不代表缺陷尺寸、深度或标准限值。", small))
    story.append(Paragraph("三、系统辅助分析与处置建议", heading))
    story.append(para("异常描述：" + str(cog.get("defect_desc") or "未记录")))
    story.append(para("处置建议：" + str(cog.get("suggestion") or "请结合原图人工复核")))
    if route in ("vlm_review", "slow", "general"):
        story.append(para("VLM输出：" + str(cog.get("vlm_conclusion") or cog.get("conclusion") or "未记录")))
    ids = cog.get("ref_ids") or []
    refs = cog.get("retrieved") or result.get("references") or []
    if ids or refs:
        story.append(Paragraph("四、知识检索与来源说明", heading))
        if any(str(ref.get('id', '')).startswith('GB34528') for ref in refs):
            story.append(para('历史引用已停用：原知识库的标准编号及页码未经核实，不能作为国标原文或质量判定依据，请用新版重新检测。'))
        else:
            story.append(para('项目知识卡为教学示例，标准目录仅核验编号与名称；本系统尚未接入标准原文，不能据此给出条文或尺寸限值。'))
        story.append(para("模型引用ID：" + "、".join(map(str, ids))))
        for ref in refs:
            location = f" / P{ref['page']}" if ref.get('page') is not None else ''
            story.append(para(f"[{ref.get('id', '—')}] {ref.get('standard', '')}{location} / {ref.get('section', '')}", small))
            story.append(para(ref.get("text", "")))
        story.append(para("检索片段与引用ID仅供核对；应人工确认原文、版本和适用范围。", small))
    def footer(canvas, document):
        canvas.setFont(_FONT_NAME, 7)
        canvas.setFillColor(colors.HexColor("#71818b"))
        canvas.drawString(1.8*cm, 1.0*cm, "MetalQC · AI自动处理 / 例外复核 / 质量追溯")
        canvas.drawRightString(A4[0]-1.8*cm, 1.0*cm, f"第 {document.page} 页")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output_path
