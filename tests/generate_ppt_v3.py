# -*- coding: utf-8 -*-
"""
开题答辩 PPT v3 - 优化美化版
在 v2 基础上：
1. 用精简版实验图替换文字过密的原图
2. 封面增加科技感装饰元素
3. 统一卡片风格，精简副标题废话，突出关键数字
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn

ROOT = r"D:\programme file\ResearchArticle"
RESULT_DIR = os.path.join(ROOT, "results")
PPT_DIR = os.path.join(RESULT_DIR, "ppt_v3")
OUT_PATH = os.path.join(ROOT, "开题答辩_双链路智能质检辅助系统.pptx")

# ===== 配色 =====
C_NAVY    = RGBColor(0x0A, 0x1F, 0x44)
C_BLUE    = RGBColor(0x15, 0x65, 0xC0)
C_BLUE_L  = RGBColor(0x42, 0xA5, 0xF5)
C_BLUE_BG = RGBColor(0xE3, 0xF2, 0xFD)
C_ORANGE  = RGBColor(0xE6, 0x51, 0x00)
C_ORANGE_BG = RGBColor(0xFF, 0xF3, 0xE0)
C_GREEN   = RGBColor(0x2E, 0x7D, 0x32)
C_GREEN_BG= RGBColor(0xE8, 0xF5, 0xE9)
C_RED     = RGBColor(0xC6, 0x28, 0x28)
C_RED_BG  = RGBColor(0xFF, 0xEB, 0xEE)
C_PURPLE  = RGBColor(0x6A, 0x1B, 0x9A)
C_PURPLE_BG = RGBColor(0xF3, 0xE5, 0xF5)
C_GRAY    = RGBColor(0x55, 0x55, 0x55)
C_GRAY_L  = RGBColor(0x9E, 0x9E, 0x9E)
C_BG      = RGBColor(0xFA, 0xFA, 0xFA)
C_WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
C_DARK    = RGBColor(0x21, 0x21, 0x21)
C_LINE    = RGBColor(0xE0, 0xE0, 0xE0)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW = prs.slide_width
SH = prs.slide_height

def blank():
    return prs.slides.add_slide(prs.slide_layouts[6])

def set_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def rect(slide, l, t, w, h, fill, line=None, line_w=1):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line; s.line.width = Pt(line_w)
    else:
        s.line.fill.background()
    return s

def rrect(slide, l, t, w, h, fill, line=None, line_w=1.5):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line; s.line.width = Pt(line_w)
    else:
        s.line.fill.background()
    return s

def txt(slide, l, t, w, h, text, size=18, bold=False, color=C_DARK,
        align=PP_ALIGN.LEFT, font="微软雅黑", anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Pt(2); tf.margin_right = Pt(2)
    tf.margin_top = Pt(1); tf.margin_bottom = Pt(1)
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size); p.font.bold = bold
    p.font.color.rgb = color; p.font.name = font
    p.alignment = align
    return tb

def multiline(slide, l, t, w, h, lines, size=14, color=C_GRAY, spacing=1.4, font="微软雅黑"):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(line, tuple):
            text, b, c = line[0], line[1] if len(line)>1 else False, line[2] if len(line)>2 else color
        else:
            text, b, c = line, False, color
        p.text = text
        p.font.size = Pt(size); p.font.bold = b
        p.font.color.rgb = c; p.font.name = font
        p.space_after = Pt(size * 0.25)
        p.line_spacing = spacing
    return tb

def img(slide, path, l, t, width=None, height=None):
    if os.path.exists(path):
        if width and height: slide.shapes.add_picture(path, l, t, width=width, height=height)
        elif width: slide.shapes.add_picture(path, l, t, width=width)
        elif height: slide.shapes.add_picture(path, l, t, height=height)
        else: slide.shapes.add_picture(path, l, t)
        return True
    return False

def page_header(slide, title, subtitle=None, num=None, accent=C_BLUE):
    """统一页面顶部：左侧标题 + 强调色条 + 页码"""
    rect(slide, 0, 0, SW, Inches(0.07), accent)
    txt(slide, Inches(0.6), Inches(0.28), Inches(10), Inches(0.55),
        title, size=26, bold=True, color=C_NAVY)
    if subtitle:
        txt(slide, Inches(0.6), Inches(0.82), Inches(10.5), Inches(0.35),
            subtitle, size=13, color=C_GRAY_L)
    if num:
        # 页码徽标
        circ = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(12.05), Inches(0.25), Inches(0.62), Inches(0.62))
        circ.fill.solid(); circ.fill.fore_color.rgb = accent
        circ.line.fill.background()
        txt(slide, Inches(12.05), Inches(0.36), Inches(0.62), Inches(0.4),
            f"{num:02d}", size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    rect(slide, Inches(0.6), Inches(1.18), Inches(12.1), Pt(1.5), C_LINE)

def big_number(slide, l, t, number, label, color=C_BLUE, num_size=44):
    """超大数字 + 标签"""
    txt(slide, l, t, Inches(3), Inches(0.85), number, size=num_size, bold=True,
        color=color, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txt(slide, l, t + Inches(0.85), Inches(3), Inches(0.35), label, size=12,
        color=C_GRAY, align=PP_ALIGN.CENTER)

def add_tech_decor(slide, color=C_BLUE_L, alpha_hex="22"):
    """科技感装饰：角落网格 + 圆点（用半透明色块近似）"""
    # 右上角装饰圆环
    for i, r in enumerate([1.0, 0.72, 0.44]):
        c = slide.shapes.add_shape(MSO_SHAPE.OVAL,
            Inches(12.6 - r*0.5), Inches(-0.25 + i*0.02), Inches(r), Inches(r))
        c.fill.background()
        c.line.color.rgb = color; c.line.width = Pt(1.2)
    # 左下角装饰
    for i, r in enumerate([0.9, 0.6]):
        c = slide.shapes.add_shape(MSO_SHAPE.OVAL,
            Inches(-0.35 + i*0.15), Inches(7.0 - r*0.5), Inches(r), Inches(r))
        c.fill.background()
        c.line.color.rgb = color; c.line.width = Pt(1.2)

def add_corner_tag(slide, text, color=C_ORANGE, l=Inches(11.0), t=Inches(6.75)):
    """右下角小标签"""
    tag = rrect(slide, l, t, Inches(2.2), Inches(0.45), C_ORANGE_BG)
    txt(slide, l, t + Inches(0.06), Inches(2.2), Inches(0.33),
        text, size=11, bold=True, color=color, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================
# 第1页：封面（深色 + 科技感装饰）
# ================================================================
s = blank()
set_bg(s, C_NAVY)
# 背景装饰：右下角大圆环 + 网格感色块
for r in [3.2, 2.4, 1.6]:
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(11.8 - (r-1.6)*0.3), Inches(5.6 - (r-1.6)*0.3), Inches(r), Inches(r))
    c.fill.background()
    c.line.color.rgb = RGBColor(0x1E, 0x3A, 0x6E); c.line.width = Pt(1)
# 右上角细线装饰
for i, wdt in enumerate([0.15, 0.1, 0.05]):
    rect(s, Inches(12.6 + i*0.18), Inches(0.3), Inches(0.04), Inches(0.6), RGBColor(0x2A, 0x4A, 0x82))
# 右侧强调竖条
rect(s, Inches(11.6), 0, Inches(1.733), SH, RGBColor(0x0D, 0x47, 0xA1))
rect(s, Inches(11.42), 0, Inches(0.14), SH, C_ORANGE)
# 顶部课程标签 + 标签底色条
rect(s, Inches(1.0), Inches(1.1), Inches(4.6), Inches(0.5), RGBColor(0x14, 0x30, 0x5E))
txt(s, Inches(1.0), Inches(1.18), Inches(4.6), Inches(0.35),
    "《人工智能应用实践》课程开题报告", size=15, color=C_BLUE_L, align=PP_ALIGN.CENTER)
# 主标题
txt(s, Inches(1.0), Inches(2.1), Inches(10), Inches(0.9),
    "面向金属表面缺陷的", size=30, bold=True, color=C_WHITE)
txt(s, Inches(1.0), Inches(2.95), Inches(10), Inches(1.1),
    "双链路门控智能质检辅助系统", size=42, bold=True, color=C_WHITE)
# 橙色分隔
rect(s, Inches(1.0), Inches(4.35), Inches(2.8), Pt(3.5), C_ORANGE)
# 一句话定位（大字）
txt(s, Inches(1.0), Inches(4.7), Inches(10), Inches(0.55),
    "置信度门控 · 少样本换产 · 引用可溯源", size=22, bold=True, color=C_BLUE_L)
# 底部信息
txt(s, Inches(1.0), Inches(6.15), Inches(6), Inches(0.35),
    "山东大学人工智能学院", size=14, color=C_GRAY_L)
txt(s, Inches(1.0), Inches(6.55), Inches(9), Inches(0.35),
    "团队：XXX / XXX / XXX    指导教师：赵玉琨 副教授    2026.09",
    size=12, color=C_GRAY_L)

# ================================================================
# 第2页：目录
# ================================================================
s = blank()
set_bg(s, C_BG)
page_header(s, "目录", "CONTENTS", 2)
add_tech_decor(s)

items = [
    ("01", "问题与需求", "行业痛点 · 系统定位"),
    ("02", "研究现状", "技术路线 · 现有不足"),
    ("03", "系统架构", "双链路门控 · 可溯源校验"),
    ("04", "创新点", "架构 · 工作流 · 工程实现"),
    ("05", "可行性与先导实验", "技术验证 · 实测数据"),
    ("06", "评估与计划", "指标体系 · 进度分工"),
]
for i, (num, title, desc) in enumerate(items):
    row = i // 2; col = i % 2
    l = Inches(0.8 + col * 6.1)
    t = Inches(1.55 + row * 1.75)
    rrect(s, l, t, Inches(5.7), Inches(1.5), C_WHITE, C_LINE, 1)
    # 左侧色块带图标感
    rect(s, l, t, Inches(1.15), Inches(1.5), C_BLUE)
    txt(s, l, t + Inches(0.3), Inches(1.15), Inches(0.9), num, size=34, bold=True,
        color=C_WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, l + Inches(1.35), t + Inches(0.28), Inches(4.2), Inches(0.5),
        title, size=20, bold=True, color=C_NAVY)
    txt(s, l + Inches(1.35), t + Inches(0.85), Inches(4.2), Inches(0.4),
        desc, size=13, color=C_GRAY)

# ================================================================
# 第3页：行业痛点
# ================================================================
s = blank()
set_bg(s, C_BG)
page_header(s, "一、待解决的问题", "中小机械加工厂 · 多品种小批量 · 离线抽检", 3)

pains = [
    ("换产成本高", "新零件需标注缺陷、训练模型", "周期长 · 成本高 · 小批量难承担", C_RED, C_RED_BG),
    ("分析效率低", "翻标准、写报告、分析根因", "依赖人工经验 · 单份报告30分钟+", C_ORANGE, C_ORANGE_BG),
    ("合规性弱", "人工判定标准不统一", "无溯源记录 · 不符合可追溯要求", C_BLUE, C_BLUE_BG),
]
for i, (title, sub, desc, c, bg) in enumerate(pains):
    l = Inches(0.6 + i * 4.15)
    t = Inches(1.55)
    rrect(s, l, t, Inches(3.85), Inches(4.35), C_WHITE, C_LINE, 1)
    rect(s, l, t, Inches(3.85), Inches(0.14), c)
    txt(s, l + Inches(0.3), t + Inches(0.35), Inches(1), Inches(0.8),
        f"0{i+1}", size=52, bold=True, color=bg)
    txt(s, l + Inches(0.3), t + Inches(1.45), Inches(3.3), Inches(0.55),
        title, size=24, bold=True, color=c)
    rect(s, l + Inches(0.3), t + Inches(2.1), Inches(1.2), Pt(2.5), c)
    txt(s, l + Inches(0.3), t + Inches(2.4), Inches(3.3), Inches(0.6),
        sub, size=15, bold=True, color=C_DARK)
    txt(s, l + Inches(0.3), t + Inches(3.1), Inches(3.3), Inches(0.8),
        desc, size=13, color=C_GRAY)

rrect(s, Inches(0.6), Inches(6.15), Inches(12.1), Inches(0.85), C_NAVY)
txt(s, Inches(0.6), Inches(6.28), Inches(12.1), Inches(0.6),
    "现有方案：纯检测无分析 · 纯大模型无定位 · 定制成本高 —— 缺一个兼顾效率、精度、合规的完整方案",
    size=15, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================
# 第4页：系统定位
# ================================================================
s = blank()
set_bg(s, C_BG)
page_header(s, "一、系统定位与目标", "人在回路的离线质检辅助工具", 4)

rrect(s, Inches(0.6), Inches(1.45), Inches(12.1), Inches(1.55), C_NAVY)
txt(s, Inches(1.0), Inches(1.62), Inches(2), Inches(0.4),
    "系统定位", size=15, bold=True, color=C_ORANGE)
txt(s, Inches(1.0), Inches(2.05), Inches(11.2), Inches(0.8),
    "AI 负责定位、初筛、整理依据，人工负责最终确认 —— 质检「副驾驶」，不替代人",
    size=21, bold=True, color=C_WHITE)

goals = [
    ("少样本检测", "仅需10-20张正常样本建库\n换产无需模型训练", C_BLUE, "01"),
    ("自动化分析", "自动匹配标准、生成报告\n降低质检人工工作量", C_GREEN, "02"),
    ("可溯源合规", "结论绑定标准原文页码\n错误可识别、可校验", C_ORANGE, "03"),
]
for i, (title, desc, c, num) in enumerate(goals):
    l = Inches(0.6 + i * 4.15)
    t = Inches(3.35)
    rrect(s, l, t, Inches(3.85), Inches(2.75), C_WHITE, c, 2)
    rect(s, l, t, Inches(3.85), Inches(0.68), c)
    txt(s, l + Inches(0.25), t + Inches(0.14), Inches(1), Inches(0.42),
        num, size=20, bold=True, color=C_WHITE)
    txt(s, l + Inches(1.2), t + Inches(0.16), Inches(2.5), Inches(0.42),
        title, size=19, bold=True, color=C_WHITE)
    txt(s, l + Inches(0.3), t + Inches(1.05), Inches(3.3), Inches(1.5),
        desc, size=15, color=C_DARK)

txt(s, Inches(0.6), Inches(6.5), Inches(12.1), Inches(0.4),
    "应用场景：金属零件表面缺陷离线抽检  ·  新产线快速质检搭建  ·  质检新人培训辅助",
    size=14, color=C_GRAY, align=PP_ALIGN.CENTER)

# ================================================================
# 第5页：研究现状
# ================================================================
s = blank()
set_bg(s, C_BG)
page_header(s, "二、研究现状", "工业缺陷检测两条技术路线 + 大模型RAG应用", 5)

rrect(s, Inches(0.6), Inches(1.45), Inches(5.9), Inches(2.35), C_WHITE, C_RED, 1.5)
rect(s, Inches(0.6), Inches(1.45), Inches(5.9), Inches(0.55), C_RED)
txt(s, Inches(0.8), Inches(1.53), Inches(5.5), Inches(0.4),
    "监督学习路线（YOLO / CNN）", size=16, bold=True, color=C_WHITE)
multiline(s, Inches(0.85), Inches(2.2), Inches(5.5), Inches(1.4),
    [("✓ 检测精度高", True, C_GREEN),
     ("✗ 需大量标注数据，训练周期长", False, C_DARK),
     ("✗ 换产重新标注成本高，少样本适配差", False, C_DARK)],
    size=13)

rrect(s, Inches(0.6), Inches(4.0), Inches(5.9), Inches(2.35), C_WHITE, C_GREEN, 1.5)
rect(s, Inches(0.6), Inches(4.0), Inches(5.9), Inches(0.55), C_GREEN)
txt(s, Inches(0.8), Inches(4.08), Inches(5.5), Inches(0.4),
    "零样本异常检测（WinCLIP等）", size=16, bold=True, color=C_WHITE)
multiline(s, Inches(0.85), Inches(4.75), Inches(5.5), Inches(1.4),
    [("✓ 仅需正常样本建库，适配小批量换产", True, C_GREEN),
     ("✓ 无需标注缺陷数据", True, C_GREEN),
     ("✗ 只有检测定位，缺少深度分析能力", False, C_DARK)],
    size=13)

rrect(s, Inches(6.8), Inches(1.45), Inches(5.9), Inches(2.35), C_WHITE, C_PURPLE, 1.5)
rect(s, Inches(6.8), Inches(1.45), Inches(5.9), Inches(0.55), C_PURPLE)
txt(s, Inches(7.0), Inches(1.53), Inches(5.5), Inches(0.4),
    "生成式VLM + RAG知识库", size=16, bold=True, color=C_WHITE)
multiline(s, Inches(7.05), Inches(2.2), Inches(5.5), Inches(1.4),
    [("✓ 能理解图像、生成描述、结合知识库问答", True, C_GREEN),
     ("✗ 多为纯文本方案，缺少与视觉检测融合", False, C_DARK),
     ("✗ 幻觉不可控，不符合工业合规要求", False, C_DARK)],
    size=13)

rrect(s, Inches(6.8), Inches(4.0), Inches(5.9), Inches(2.35), C_NAVY)
txt(s, Inches(7.0), Inches(4.15), Inches(5.5), Inches(0.4),
    "现有方案共性不足", size=16, bold=True, color=C_ORANGE)
multiline(s, Inches(7.05), Inches(4.7), Inches(5.5), Inches(1.4),
    ["要么只有检测没有分析", "要么只有分析没有定位",
     ("缺少兼顾效率、精度、合规性的完整工作流", True, C_WHITE)],
    size=14, color=C_WHITE, spacing=1.6)

# ================================================================
# 第6页：系统架构 · 工作流总览（原生形状工作流图）
# ================================================================
s = blank()
set_bg(s, C_BG)
page_header(s, "三、系统架构 · 工作流总览",
            "数据输入 → 感知检测 → 门控路由 → 双链路分析 → 人工确认", 6)

def _arrow(slide, x1, y1, x2, y2, color=C_GRAY_L, w=1.8, dash=False):
    cn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    cn.line.color.rgb = color; cn.line.width = Pt(w)
    ln = cn.line._get_or_add_ln()
    if dash:
        ln.append(ln.makeelement(qn('a:prstDash'), {'val': 'dash'}))
    ln.append(ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'}))
    return cn

CX = Inches(6.667)

# —— 第1层：数据输入（三框） ——
inputs = ["正常样本图", "待检工件图", "质检标准文档"]
bw = Inches(2.7); bh = Inches(0.5); g = Inches(0.3)
x0 = (SW - (bw * 3 + g * 2)) // 2
y1 = Inches(1.26)
for i, name in enumerate(inputs):
    x = x0 + i * (bw + g)
    rrect(s, x, y1, bw, bh, C_WHITE, C_GRAY, 1.2)
    txt(s, x, y1, bw, bh, name, size=13, bold=True, color=C_NAVY,
        align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
txt(s, Inches(0.45), y1, Inches(1.25), bh, "数据输入", size=11, bold=True,
    color=C_GRAY, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)

# —— 第2层：感知检测 ——
y2 = Inches(1.98); h2 = Inches(0.6); x2 = Inches(1.85); w2 = Inches(9.63)
_arrow(s, CX, y1 + bh, CX, y2)
rrect(s, x2, y2, w2, h2, C_BLUE)
txt(s, x2, y2, w2, h2,
    "感知检测层 · WinCLIP：特征库建库    |    多尺度滑窗    |    异常热力图    |    全局异常置信度",
    size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# —— 第3层：门控菱形 ——
y3 = Inches(2.86); dw = Inches(3.6); dh = Inches(0.84)
dx = CX - dw // 2
_arrow(s, CX, y2 + h2, CX, y3)
dia = s.shapes.add_shape(MSO_SHAPE.DIAMOND, dx, y3, dw, dh)
dia.fill.solid(); dia.fill.fore_color.rgb = C_ORANGE; dia.line.fill.background()
dtf = dia.text_frame; dtf.word_wrap = True; dtf.vertical_anchor = MSO_ANCHOR.MIDDLE
dp = dtf.paragraphs[0]; dp.alignment = PP_ALIGN.CENTER
dr = dp.add_run(); dr.text = "门控路由层"; dr.font.size = Pt(12); dr.font.bold = True
dr.font.color.rgb = C_WHITE; dr.font.name = "微软雅黑"
dp2 = dtf.add_paragraph(); dp2.alignment = PP_ALIGN.CENTER
dr2 = dp2.add_run(); dr2.text = "分数 ≥ 0.2243 ?"; dr2.font.size = Pt(11)
dr2.font.color.rgb = C_WHITE; dr2.font.name = "微软雅黑"
txt(s, dx - Inches(1.5), y3 + Inches(0.22), Inches(1.4), Inches(0.4),
    "是 · 高置信", size=12, bold=True, color=C_GREEN,
    align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
txt(s, dx + dw + Inches(0.1), y3 + Inches(0.22), Inches(1.45), Inches(0.4),
    "否 · 低置信", size=12, bold=True, color=C_PURPLE,
    align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)

# —— 第4层：快慢双分支 ——
y4 = Inches(4.0); h4 = Inches(1.1)
fx = Inches(0.9); fw = Inches(5.55)
sx = Inches(6.88); sw4 = Inches(5.55)
_arrow(s, dx + Inches(0.45), y3 + dh, fx + fw // 2, y4, color=C_GREEN, w=2)
_arrow(s, dx + dw - Inches(0.45), y3 + dh, sx + sw4 // 2, y4, color=C_PURPLE, w=2)
# 快链路
rrect(s, fx, y4, fw, h4, C_GREEN_BG, C_GREEN, 2)
rect(s, fx, y4, fw, Inches(0.42), C_GREEN)
txt(s, fx, y4 + Inches(0.03), fw, Inches(0.36), "快链路 · 高置信直出",
    size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
txt(s, fx + Inches(0.2), y4 + Inches(0.5), fw - Inches(0.4), Inches(0.36),
    "异常热力图 → 缺陷定位 → 等级判定", size=13, bold=True,
    color=C_NAVY, align=PP_ALIGN.CENTER)
txt(s, fx + Inches(0.2), y4 + Inches(0.82), fw - Inches(0.4), Inches(0.28),
    "耗时 2-3 秒 · 省算力", size=11, bold=True, color=C_ORANGE, align=PP_ALIGN.CENTER)
# 慢链路
rrect(s, sx, y4, sw4, h4, C_PURPLE_BG, C_PURPLE, 2)
rect(s, sx, y4, sw4, Inches(0.42), C_PURPLE)
txt(s, sx, y4 + Inches(0.03), sw4, Inches(0.36), "慢链路 · 低置信精析（认知层）",
    size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
txt(s, sx + Inches(0.15), y4 + Inches(0.5), sw4 - Inches(0.3), Inches(0.36),
    "裁剪 → VLM描述 → RAG标准检索 → 引用校验 → 结构化报告",
    size=12, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
txt(s, sx + Inches(0.2), y4 + Inches(0.82), sw4 - Inches(0.4), Inches(0.28),
    "耗时 20-30 秒 · 结论可溯源", size=11, bold=True, color=C_ORANGE, align=PP_ALIGN.CENTER)

# —— 第5层：汇聚 + 人工确认 ——
y5 = Inches(5.36); h5 = Inches(0.56); x5 = Inches(1.85); w5 = Inches(9.63)
_arrow(s, fx + fw // 2, y4 + h4, x5 + int(w5 * 0.26), y5, color=C_GREEN, w=2)
_arrow(s, sx + sw4 // 2, y4 + h4, x5 + int(w5 * 0.74), y5, color=C_PURPLE, w=2)
rrect(s, x5, y5, w5, h5, C_NAVY)
txt(s, x5, y5, w5, h5,
    "分析结果与质检报告    →    人工最终确认（人在回路，AI 不替代人）",
    size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# —— 反馈回路（左侧橙色虚线） ——
fbx = Inches(0.42)
_arrow(s, x5, y5 + h5 // 2, fbx, y5 + h5 // 2, color=C_ORANGE, w=1.5, dash=True)
_arrow(s, fbx, y5 + h5 // 2, fbx, y2 + h5 // 2, color=C_ORANGE, w=1.5, dash=True)
_arrow(s, fbx, y2 + h5 // 2, x2, y2 + h5 // 2, color=C_ORANGE, w=1.5, dash=True)

# —— 反馈说明条 ——
y6 = Inches(6.12); h6 = Inches(0.44)
rrect(s, Inches(0.9), y6, Inches(11.53), h6, C_ORANGE_BG)
txt(s, Inches(0.9), y6, Inches(11.53), h6,
    "反馈回路：人工复核结果 → 案例库沉淀 → 阈值辅助标定 / 标准库审核更新",
    size=12, bold=True, color=C_ORANGE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# —— 底部：知识库底座 ——
y7 = Inches(6.74); h7 = Inches(0.52)
rrect(s, Inches(0.9), y7, Inches(11.53), h7, C_NAVY)
txt(s, Inches(0.9), y7, Inches(11.53), h7,
    "知识库底座：正常特征库  |  质检标准库  |  历史案例库  |  报告模板库",
    size=13, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================
# 第7页：核心机制 · 双链路门控（新精简大图）
# ================================================================
s = blank()
set_bg(s, C_BG)
page_header(s, "三、核心机制 · 双链路门控", "高置信快检 · 低置信精析 · 置信度智能路由", 7)

# 左侧要点
points = [
    ("感知层统一计算", "WinCLIP 像素级异常分数 → 全局置信度", C_BLUE),
    ("高置信 → 快链路", "分数≥阈值，直接输出定位/等级/热力图", C_GREEN),
    ("低置信 → 慢链路", "分数<阈值，VLM描述 → RAG匹配 → 报告", C_ORANGE),
    ("三级反馈反哺", "案例库 → 阈值调整 → 标准库人工审核", C_PURPLE),
]
for i, (title, desc, c) in enumerate(points):
    t = Inches(1.4 + i * 1.32)
    rrect(s, Inches(0.6), t, Inches(4.0), Inches(1.12), C_WHITE, c, 1.5)
    rect(s, Inches(0.6), t, Inches(0.1), Inches(1.12), c)
    txt(s, Inches(0.85), t + Inches(0.12), Inches(3.6), Inches(0.4),
        title, size=15, bold=True, color=c)
    txt(s, Inches(0.85), t + Inches(0.55), Inches(3.6), Inches(0.5),
        desc, size=11, color=C_GRAY)

rrect(s, Inches(0.6), Inches(6.8), Inches(4.0), Inches(0.45), C_ORANGE_BG)
txt(s, Inches(0.7), Inches(6.83), Inches(3.8), Inches(0.4),
    "阈值：中位数+1.5×MAD = 0.2243", size=12, bold=True, color=C_ORANGE,
    anchor=MSO_ANCHOR.MIDDLE)

# 右侧：精简版大图
img(s, os.path.join(PPT_DIR, "dual_path_v3.png"),
    Inches(4.75), Inches(1.35), width=Inches(8.35))

# ================================================================
# 第8页：核心机制 · 页码级可溯源校验（新对比大图）
# ================================================================
s = blank()
set_bg(s, C_BG)
page_header(s, "三、核心机制 · 页码级可溯源校验", "不追求零幻觉，追求引用可校验、错误可识别", 8)

# 左侧：5步机制
steps = [
    "片段ID化：标准片段绑定原文+页码+标准号",
    "RAG检索：返回片段+引用ID+页码",
    "生成约束：Prompt强制VLM输出引用ID",
    "后处理校验：检查ID是否在检索结果中",
    "降级模式：校验失败 → 展示原文供人工选",
]
for i, step in enumerate(steps):
    t = Inches(1.38 + i * 1.0)
    rrect(s, Inches(0.6), t, Inches(3.8), Inches(0.85), C_WHITE, C_GREEN, 1)
    circ = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.75), t + Inches(0.18), Inches(0.5), Inches(0.5))
    circ.fill.solid(); circ.fill.fore_color.rgb = C_GREEN
    circ.line.fill.background()
    txt(s, Inches(0.75), t + Inches(0.2), Inches(0.5), Inches(0.45),
        str(i + 1), size=16, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, Inches(1.4), t + Inches(0.12), Inches(2.9), Inches(0.65),
        step, size=11, color=C_DARK, anchor=MSO_ANCHOR.MIDDLE)

# 右侧：精简对比图
img(s, os.path.join(PPT_DIR, "validation_v3.png"),
    Inches(4.7), Inches(1.6), width=Inches(8.35))

# ================================================================
# 第9页：创新点
# ================================================================
s = blank()
set_bg(s, C_BG)
page_header(s, "四、项目创新点", "底层模型用开源，创新集中在应用架构与工程落地", 9)

innovations = [
    ("应用架构创新", "置信度门控的双链路质检架构",
     "高置信快检、低置信精析，兼顾效率与深度；避免全量图走大模型的算力浪费",
     C_BLUE, C_BLUE_BG),
    ("场景工作流创新", "少样本全流程质检闭环",
     "建库→检测→定位→分析→报告→迭代；仅需10-20张正常样本，换产0.5-1天",
     C_GREEN, C_GREEN_BG),
    ("工程实现创新", "引用ID绑定的可溯源校验机制",
     "所有判定绑定标准原文页码，输出自动校验；失败自动降级，符合工业合规",
     C_ORANGE, C_ORANGE_BG),
]
for i, (cat, title, desc, c, bg) in enumerate(innovations):
    l = Inches(0.6 + i * 4.15)
    t = Inches(1.45)
    rrect(s, l, t, Inches(3.85), Inches(4.55), C_WHITE, c, 2)
    rect(s, l, t, Inches(3.85), Inches(1.0), c)
    txt(s, l + Inches(0.25), t + Inches(0.15), Inches(3.4), Inches(0.35),
        cat, size=14, bold=True, color=bg)
    txt(s, l + Inches(0.25), t + Inches(0.5), Inches(3.4), Inches(0.45),
        title, size=18, bold=True, color=C_WHITE)
    txt(s, l + Inches(0.2), t + Inches(1.2), Inches(2), Inches(1.0),
        f"0{i+1}", size=64, bold=True, color=bg)
    txt(s, l + Inches(0.3), t + Inches(2.5), Inches(3.3), Inches(1.8),
        desc, size=14, color=C_DARK)

rrect(s, Inches(0.6), Inches(6.3), Inches(12.1), Inches(0.7), C_NAVY)
txt(s, Inches(0.6), Inches(6.38), Inches(12.1), Inches(0.55),
    "技术边界：底层视觉模型、语言模型、RAG框架均采用开源成熟组件，不做模型层原创",
    size=14, bold=True, color=C_ORANGE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================
# 第10页：技术可行性（超大数字）
# ================================================================
s = blank()
set_bg(s, C_BG)
page_header(s, "五、可行性分析 · 技术与数据", "MVTec-AD metal_nut 零样本无训练实测", 10)

metrics = [
    ("0.9868", "检测 AUROC", C_BLUE),
    ("94.78%", "准确率", C_GREEN),
    ("96.77%", "精确率 / 召回率", C_ORANGE),
    ("0.2243", "门控阈值", C_PURPLE),
]
for i, (num, label, c) in enumerate(metrics):
    row = i // 2; col = i % 2
    l = Inches(0.6 + col * 2.9)
    t = Inches(1.45 + row * 2.35)
    rrect(s, l, t, Inches(2.7), Inches(2.1), C_WHITE, c, 2)
    txt(s, l, t + Inches(0.35), Inches(2.7), Inches(1.0), num, size=42, bold=True,
        color=c, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, l, t + Inches(1.4), Inches(2.7), Inches(0.4), label, size=13,
        color=C_GRAY, align=PP_ALIGN.CENTER)

rrect(s, Inches(6.6), Inches(1.45), Inches(6.2), Inches(4.7), C_WHITE, C_LINE, 1)
txt(s, Inches(6.8), Inches(1.6), Inches(5.8), Inches(0.4),
    "正常 vs 缺陷 异常分数分布", size=16, bold=True, color=C_NAVY)
img(s, os.path.join(RESULT_DIR, "perception", "score_distribution.png"),
    Inches(6.9), Inches(2.15), width=Inches(5.6))

rrect(s, Inches(0.6), Inches(6.35), Inches(12.1), Inches(0.7), C_BLUE_BG)
txt(s, Inches(0.8), Inches(6.42), Inches(11.7), Inches(0.55),
    "数据：MVTec-AD 公开数据集（115张 = 93缺陷 + 22正常） | 算力：CPU 可跑（感知2-3秒/图，认知20-30秒/图） | 无需企业私有数据",
    size=12, color=C_DARK, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================
# 第11页：先导实验（2x2 拼版，用精简图）
# ================================================================
s = blank()
set_bg(s, C_BG)
page_header(s, "五、先导实验 · 实测结果", "核心技术路线可行性已验证，可进入中期开发", 11)

experiments = [
    ("感知层 · 高置信快链路", "high_conf_sample.png", "perception", C_GREEN,
     "异常分数 0.47 ≥ 阈值 0.22，直接输出缺陷定位"),
    ("感知层 · 低置信触发慢链路", "low_conf_sample.png", "perception", C_ORANGE,
     "异常分数 ≈ 阈值，触发 VLM+RAG 深度分析"),
    ("认知层 · RAG 标准检索", "retrieval_h.png", "ppt_v3", C_BLUE,
     "TF-IDF 检索 Top-3，返回引用ID + 页码"),
    ("认知层 · 四步完整流程", "cognition_v3.png", "ppt_v3", C_PURPLE,
     "描述 → 检索 → 校验 → 结构化输出"),
]
for i, (title, fname, sub, c, desc) in enumerate(experiments):
    row = i // 2; col = i % 2
    l = Inches(0.6 + col * 6.15)
    t = Inches(1.4 + row * 2.85)
    rrect(s, l, t, Inches(5.95), Inches(2.7), C_WHITE, C_LINE, 1)
    rect(s, l, t, Inches(5.95), Inches(0.45), c)
    txt(s, l + Inches(0.15), t + Inches(0.05), Inches(5.0), Inches(0.35),
        title, size=14, bold=True, color=C_WHITE, anchor=MSO_ANCHOR.MIDDLE)
    # 图片：按比例放入，统一高度
    p = os.path.join(RESULT_DIR, sub, fname)
    if os.path.exists(p):
        from PIL import Image as PILImage
        im = PILImage.open(p)
        iw, ih = im.size
        # 目标高度 1.75 英寸，宽按比例
        target_h = Inches(1.75)
        target_w = Emu(int(target_h / ih * iw))
        max_w = Inches(5.7)
        if target_w > max_w:
            target_w = max_w
            target_h = Emu(int(target_w / iw * ih))
        s.shapes.add_picture(p, l + (Inches(5.95) - target_w) // 2,
                             t + Inches(0.55), width=target_w, height=target_h)
    txt(s, l + Inches(0.15), t + Inches(2.32), Inches(5.65), Inches(0.35),
        desc, size=11, color=C_GRAY)

# ================================================================
# 第12页：评估指标与对比实验
# ================================================================
s = blank()
set_bg(s, C_BG)
page_header(s, "六、评估指标与对比实验", "三层量化体系 + 分阶段对比", 12)

txt(s, Inches(0.6), Inches(1.4), Inches(5), Inches(0.4),
    "三层量化评估体系", size=18, bold=True, color=C_NAVY)

metric_layers = [
    ("检测层", "AUROC · 最佳F1 · 定位IoU\n误报率 · 漏报率", C_BLUE),
    ("认知层", "标准匹配Top-1准确率 · 引用准确率\n缺陷描述召回率 · 幻觉率", C_GREEN),
    ("系统层", "单图平均处理耗时 · 报告生成耗时\n人工操作步骤减少率", C_ORANGE),
]
for i, (name, metrics, c) in enumerate(metric_layers):
    t = Inches(1.95 + i * 1.5)
    rrect(s, Inches(0.6), t, Inches(5.8), Inches(1.3), C_WHITE, c, 1.5)
    rect(s, Inches(0.6), t, Inches(1.4), Inches(1.3), c)
    txt(s, Inches(0.6), t + Inches(0.35), Inches(1.4), Inches(0.6),
        name, size=18, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, Inches(2.2), t + Inches(0.2), Inches(4.0), Inches(1.0),
        metrics, size=13, color=C_DARK, anchor=MSO_ANCHOR.MIDDLE)

txt(s, Inches(6.8), Inches(1.4), Inches(5), Inches(0.4),
    "分阶段对比实验", size=18, bold=True, color=C_NAVY)

comparisons = [
    ("中期", "纯WinCLIP vs 双链路", "验证双链路的分析价值", C_BLUE),
    ("中期", "纯VLM vs 双链路", "验证感知前置的精度提升", C_GREEN),
    ("结题", "无门控串行 vs 双链路", "验证门控的效率价值", C_ORANGE),
]
for i, (stage, compare, purpose, c) in enumerate(comparisons):
    t = Inches(1.95 + i * 1.5)
    rrect(s, Inches(6.8), t, Inches(6.0), Inches(1.3), C_WHITE, c, 1.5)
    rrect(s, Inches(6.95), t + Inches(0.2), Inches(0.9), Inches(0.4), c)
    txt(s, Inches(6.95), t + Inches(0.22), Inches(0.9), Inches(0.36),
        stage, size=13, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, Inches(8.0), t + Inches(0.18), Inches(4.6), Inches(0.4),
        compare, size=15, bold=True, color=C_DARK)
    txt(s, Inches(8.0), t + Inches(0.65), Inches(4.6), Inches(0.4),
        f"目的：{purpose}", size=12, color=C_GRAY)

rrect(s, Inches(0.6), Inches(6.5), Inches(12.1), Inches(0.55), C_BLUE_BG)
txt(s, Inches(0.8), Inches(6.55), Inches(11.7), Inches(0.45),
    "对比维度：纯WinCLIP比检测层+系统层 | 纯VLM比认知层+系统层 | 无门控串行比系统层耗时",
    size=12, color=C_DARK, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================
# 第13页：进度计划与团队分工
# ================================================================
s = blank()
set_bg(s, C_BG)
page_header(s, "六、进度计划与团队分工", "12周三阶段交付，严格对应课程节点", 13)

phases = [
    ("第4周", "开题答辩", "最小Demo + PPT + 报告", "15%", C_BLUE),
    ("第9-10周", "中期检查", "核心功能Demo + 对比初测", "25%", C_GREEN),
    ("第16周", "结题答辩", "完整系统 + 报告 + 演示视频", "60%", C_ORANGE),
]
rect(s, Inches(1.5), Inches(2.0), Inches(10.3), Pt(3), C_LINE)
for i, (time, name, deliver, ratio, c) in enumerate(phases):
    cx = Inches(2.0 + i * 4.5)
    dot = s.shapes.add_shape(MSO_SHAPE.OVAL, cx - Inches(0.15), Inches(1.85), Inches(0.3), Inches(0.3))
    dot.fill.solid(); dot.fill.fore_color.rgb = c; dot.line.fill.background()
    rrect(s, cx - Inches(1.8), Inches(2.4), Inches(3.6), Inches(2.0), C_WHITE, c, 1.5)
    rect(s, cx - Inches(1.8), Inches(2.4), Inches(3.6), Inches(0.5), c)
    txt(s, cx - Inches(1.7), Inches(2.45), Inches(3.4), Inches(0.4),
        f"{time} · {name}", size=15, bold=True, color=C_WHITE, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, cx - Inches(1.6), Inches(3.05), Inches(3.2), Inches(0.6),
        deliver, size=12, color=C_DARK)
    txt(s, cx - Inches(1.6), Inches(3.7), Inches(3.2), Inches(0.5),
        f"考核占比 {ratio}", size=16, bold=True, color=c)

txt(s, Inches(0.6), Inches(4.8), Inches(5), Inches(0.4),
    "团队分工（3人）", size=18, bold=True, color=C_NAVY)

roles = [
    ("项目负责人", "总架构 · 认知层 · 文档总纂 · 答辩主讲", C_BLUE),
    ("感知算法", "WinCLIP部署 · 多尺度优化 · 指标测试", C_GREEN),
    ("系统支持", "界面开发 · 文档排版 · 演示保障", C_ORANGE),
]
for i, (role, duty, c) in enumerate(roles):
    l = Inches(0.6 + i * 4.15)
    rrect(s, l, Inches(5.35), Inches(3.85), Inches(1.5), C_WHITE, c, 1.5)
    rect(s, l, Inches(5.35), Inches(0.12), Inches(1.5), c)
    txt(s, l + Inches(0.3), Inches(5.5), Inches(3.4), Inches(0.4),
        role, size=17, bold=True, color=c)
    txt(s, l + Inches(0.3), Inches(6.0), Inches(3.4), Inches(0.7),
        duty, size=13, color=C_DARK)

# ================================================================
# 第14页：风险预案（新精简图）
# ================================================================
s = blank()
set_bg(s, C_BG)
page_header(s, "七、风险预案", "每条风险都有明确的降级方案", 14)

# 直接嵌入精简风险图（按高度适配，防止底部超画布）
img(s, os.path.join(PPT_DIR, "risks_v3.png"),
    Inches(0.6), Inches(1.38), height=Inches(5.9))

# ================================================================
# 第15页：致谢
# ================================================================
s = blank()
set_bg(s, C_NAVY)
# 装饰圆环
for r in [3.0, 2.2, 1.4]:
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(11.0 - (r-1.4)*0.3), Inches(5.2 - (r-1.4)*0.3), Inches(r), Inches(r))
    c.fill.background()
    c.line.color.rgb = RGBColor(0x1E, 0x3A, 0x6E); c.line.width = Pt(1)
rect(s, 0, Inches(3.2), SW, Pt(2), C_ORANGE)
rect(s, Inches(5.5), Inches(2.8), Inches(2.3), Pt(3), C_ORANGE)

txt(s, Inches(1), Inches(1.5), Inches(11.3), Inches(1.0),
    "感谢各位老师聆听", size=44, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
txt(s, Inches(1), Inches(2.5), Inches(11.3), Inches(0.5),
    "恳请批评指正", size=24, color=C_BLUE_L, align=PP_ALIGN.CENTER)

txt(s, Inches(1), Inches(4.0), Inches(11.3), Inches(0.5),
    "面向金属表面缺陷的双链路门控智能质检辅助系统",
    size=18, color=C_WHITE, align=PP_ALIGN.CENTER)
txt(s, Inches(1), Inches(4.6), Inches(11.3), Inches(0.4),
    "山东大学人工智能学院  ·  2026.09",
    size=14, color=C_GRAY_L, align=PP_ALIGN.CENTER)

# 保存
prs.save(OUT_PATH)
print(f"PPT v3 已生成: {OUT_PATH}")
print(f"总页数: {len(prs.slides)}")
