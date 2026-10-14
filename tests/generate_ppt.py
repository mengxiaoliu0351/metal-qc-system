# -*- coding: utf-8 -*-
"""
开题答辩 PPT 生成脚本
15页，16:9，学术风格，包含所有实验结果图
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = r"D:\programme file\ResearchArticle"
RESULT_DIR = os.path.join(ROOT, "results")
OUT_PATH = os.path.join(ROOT, "开题答辩_双链路智能质检辅助系统.pptx")

# 颜色方案
COLOR_PRIMARY = RGBColor(0x15, 0x65, 0xC0)    # 主蓝
COLOR_DARK = RGBColor(0x0D, 0x47, 0xA1)       # 深蓝
COLOR_ACCENT = RGBColor(0xE6, 0x51, 0x00)     # 强调橙
COLOR_GREEN = RGBColor(0x2E, 0x7D, 0x32)      # 绿
COLOR_RED = RGBColor(0xC6, 0x28, 0x28)        # 红
COLOR_GRAY = RGBColor(0x55, 0x55, 0x55)       # 灰
COLOR_LIGHT_GRAY = RGBColor(0xF5, 0xF5, 0xF5) # 浅灰
COLOR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
COLOR_BLACK = RGBColor(0x21, 0x21, 0x21)

# 创建PPT，16:9
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SLIDE_W = prs.slide_width
SLIDE_H = prs.slide_height

def add_blank_slide():
    return prs.slides.add_slide(prs.slide_layouts[6])  # 空白布局

def add_textbox(slide, left, top, width, height, text, font_size=18,
                 bold=False, color=COLOR_BLACK, alignment=PP_ALIGN.LEFT,
                 font_name="微软雅黑"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return txBox

def add_multiline_textbox(slide, left, top, width, height, lines, font_size=16,
                           color=COLOR_BLACK, line_spacing=1.5, font_name="微软雅黑"):
    """lines: list of (text, bold, color) or just str"""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        if isinstance(line, tuple):
            text, bold, c = line[0], line[1] if len(line) > 1 else False, line[2] if len(line) > 2 else color
        else:
            text, bold, c = line, False, color
        p.text = text
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = c
        p.font.name = font_name
        p.space_after = Pt(font_size * 0.3)
    return txBox

def add_rect(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

def add_rounded_rect(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    return shape

def add_page_header(slide, title, page_num=None):
    """统一页面顶部标题栏"""
    add_rect(slide, 0, 0, SLIDE_W, Inches(0.9), COLOR_PRIMARY)
    add_textbox(slide, Inches(0.5), Inches(0.15), Inches(11), Inches(0.6),
                title, font_size=26, bold=True, color=COLOR_WHITE)
    if page_num:
        add_textbox(slide, Inches(12.2), Inches(0.25), Inches(0.8), Inches(0.4),
                    str(page_num), font_size=16, color=COLOR_WHITE, alignment=PP_ALIGN.RIGHT)
    # 底部装饰线
    add_rect(slide, 0, Inches(7.3), SLIDE_W, Inches(0.2), COLOR_LIGHT_GRAY)

def add_image_safe(slide, img_path, left, top, width=None, height=None):
    """安全插入图片，文件不存在时跳过"""
    if os.path.exists(img_path):
        if width and height:
            slide.shapes.add_picture(img_path, left, top, width=width, height=height)
        elif width:
            slide.shapes.add_picture(img_path, left, top, width=width)
        elif height:
            slide.shapes.add_picture(img_path, left, top, height=height)
        else:
            slide.shapes.add_picture(img_path, left, top)
        return True
    return False

# ============================================================
# 第1页：封面
# ============================================================
slide = add_blank_slide()
# 背景
add_rect(slide, 0, 0, SLIDE_W, SLIDE_H, COLOR_DARK)
# 装饰条
add_rect(slide, 0, Inches(2.8), SLIDE_W, Inches(0.05), COLOR_ACCENT)
# 标题
add_textbox(slide, Inches(1), Inches(1.5), Inches(11.3), Inches(1.2),
            "面向金属表面缺陷的", font_size=36, bold=True, color=COLOR_WHITE,
            alignment=PP_ALIGN.CENTER)
add_textbox(slide, Inches(1), Inches(2.2), Inches(11.3), Inches(1.2),
            "双链路门控智能质检辅助系统", font_size=44, bold=True, color=COLOR_WHITE,
            alignment=PP_ALIGN.CENTER)
# 副标题
add_textbox(slide, Inches(1), Inches(3.2), Inches(11.3), Inches(0.6),
            "《人工智能应用实践》课程开题报告", font_size=22, color=RGBColor(0xBB,0xDE,0xFB),
            alignment=PP_ALIGN.CENTER)
# 底部信息
add_textbox(slide, Inches(1), Inches(5.5), Inches(11.3), Inches(0.5),
            "山东大学人工智能学院", font_size=18, color=COLOR_WHITE, alignment=PP_ALIGN.CENTER)
add_textbox(slide, Inches(1), Inches(6.0), Inches(11.3), Inches(0.5),
            "团队成员：XXX / XXX / XXX    指导教师：赵玉琨 副教授", font_size=16,
            color=RGBColor(0x90,0xCA,0xF9), alignment=PP_ALIGN.CENTER)
add_textbox(slide, Inches(1), Inches(6.5), Inches(11.3), Inches(0.5),
            "2026年9月", font_size=16, color=RGBColor(0x90,0xCA,0xF9), alignment=PP_ALIGN.CENTER)

# ============================================================
# 第2页：目录
# ============================================================
slide = add_blank_slide()
add_page_header(slide, "目录", 2)
items = [
    ("01", "待解决的问题与需求分析"),
    ("02", "研究现状与调研分析"),
    ("03", "系统架构与核心机制"),
    ("04", "项目创新点"),
    ("05", "可行性分析与先导实验"),
    ("06", "评估指标与进度计划"),
    ("07", "团队分工与风险预案"),
]
for i, (num, title) in enumerate(items):
    row = i // 2
    col = i % 2
    left = Inches(1.0 + col * 5.8)
    top = Inches(1.5 + row * 1.5)
    add_rounded_rect(slide, left, top, Inches(5.2), Inches(1.1), COLOR_LIGHT_GRAY, COLOR_PRIMARY)
    add_textbox(slide, left + Inches(0.2), top + Inches(0.15), Inches(0.8), Inches(0.8),
                num, font_size=32, bold=True, color=COLOR_PRIMARY)
    add_textbox(slide, left + Inches(1.1), top + Inches(0.3), Inches(3.9), Inches(0.6),
                title, font_size=18, bold=True, color=COLOR_DARK)

# ============================================================
# 第3页：行业痛点
# ============================================================
slide = add_blank_slide()
add_page_header(slide, "一、待解决的问题 · 行业痛点", 3)
add_textbox(slide, Inches(0.5), Inches(1.1), Inches(12), Inches(0.5),
            "聚焦中小机械加工厂金属零件多品种小批量离线抽检场景", font_size=18,
            bold=True, color=COLOR_ACCENT)

pains = [
    ("换产成本高", "新零件质检需标注缺陷、训练模型，周期长、成本高，小批量产线难以承担", COLOR_RED),
    ("分析效率低", "翻标准、写报告、分析根因高度依赖人工经验，单份报告耗时30分钟以上", COLOR_ACCENT),
    ("合规性弱", "人工判定标准不统一，结论无溯源记录，不符合质检可追溯要求", COLOR_DARK),
]
for i, (title, desc, color) in enumerate(pains):
    left = Inches(0.5 + i * 4.2)
    top = Inches(1.9)
    add_rounded_rect(slide, left, top, Inches(3.9), Inches(3.5), COLOR_LIGHT_GRAY, color)
    add_rect(slide, left, top, Inches(3.9), Inches(0.7), color)
    add_textbox(slide, left + Inches(0.2), top + Inches(0.12), Inches(3.5), Inches(0.5),
                f"痛点{i+1}：{title}", font_size=20, bold=True, color=COLOR_WHITE)
    add_textbox(slide, left + Inches(0.25), top + Inches(1.0), Inches(3.4), Inches(2.3),
                desc, font_size=16, color=COLOR_GRAY)

add_textbox(slide, Inches(0.5), Inches(5.8), Inches(12), Inches(1.0),
            "现有方案共性不足：纯检测工具功能单一；纯大模型方案幻觉不可控；定制系统成本高，中小工厂难以复用",
            font_size=16, bold=True, color=COLOR_DARK, alignment=PP_ALIGN.CENTER)

# ============================================================
# 第4页：系统定位与目标
# ============================================================
slide = add_blank_slide()
add_page_header(slide, "一、待解决的问题 · 系统定位与目标", 4)

# 定位
add_rounded_rect(slide, Inches(0.5), Inches(1.2), Inches(12.3), Inches(1.2),
                 RGBColor(0xE3,0xF2,0xFD), COLOR_PRIMARY)
add_textbox(slide, Inches(0.8), Inches(1.35), Inches(2), Inches(0.5),
            "系统定位", font_size=20, bold=True, color=COLOR_PRIMARY)
add_textbox(slide, Inches(0.8), Inches(1.8), Inches(11.5), Inches(0.5),
            "人在回路的离线质检辅助工具，不替代人工最终质量判定；AI负责定位、初筛、整理依据，人工负责最终确认",
            font_size=16, color=COLOR_DARK)

# 三个目标
goals = [
    ("少样本检测", "少样本条件下完成缺陷检测定位，换产无需模型训练"),
    ("自动化分析", "自动匹配标准、生成分析报告，降低质检人工工作量"),
    ("可溯源合规", "所有结论可溯源、可校验，符合质检合规要求"),
]
for i, (title, desc) in enumerate(goals):
    left = Inches(0.5 + i * 4.2)
    top = Inches(2.8)
    add_rounded_rect(slide, left, top, Inches(3.9), Inches(2.0), COLOR_WHITE, COLOR_GREEN)
    add_textbox(slide, left + Inches(0.2), top + Inches(0.2), Inches(3.5), Inches(0.5),
                f"目标{i+1}：{title}", font_size=18, bold=True, color=COLOR_GREEN)
    add_textbox(slide, left + Inches(0.25), top + Inches(0.8), Inches(3.4), Inches(1.0),
                desc, font_size=15, color=COLOR_GRAY)

# 应用场景
add_textbox(slide, Inches(0.5), Inches(5.2), Inches(12), Inches(0.5),
            "应用场景", font_size=20, bold=True, color=COLOR_ACCENT)
add_textbox(slide, Inches(0.5), Inches(5.7), Inches(12), Inches(1.2),
            "金属零件表面缺陷离线抽检  ·  新产线快速质检搭建  ·  质检新人培训辅助",
            font_size=18, color=COLOR_DARK, alignment=PP_ALIGN.CENTER)

# ============================================================
# 第5页：研究现状
# ============================================================
slide = add_blank_slide()
add_page_header(slide, "二、研究现状与调研分析", 5)

# 两条技术路线
add_textbox(slide, Inches(0.5), Inches(1.1), Inches(6), Inches(0.5),
            "工业缺陷检测两条技术路线", font_size=20, bold=True, color=COLOR_PRIMARY)

add_rounded_rect(slide, Inches(0.5), Inches(1.7), Inches(5.8), Inches(2.2),
                 RGBColor(0xFF,0xF3,0xE0), COLOR_ACCENT)
add_textbox(slide, Inches(0.7), Inches(1.85), Inches(5.4), Inches(0.5),
            "监督学习路线（YOLO、CNN）", font_size=17, bold=True, color=COLOR_ACCENT)
add_textbox(slide, Inches(0.75), Inches(2.4), Inches(5.3), Inches(1.4),
            "精度高，但需大量标注数据，换产成本高，少样本场景适配差",
            font_size=15, color=COLOR_GRAY)

add_rounded_rect(slide, Inches(0.5), Inches(4.1), Inches(5.8), Inches(2.2),
                 RGBColor(0xE8,0xF5,0xE9), COLOR_GREEN)
add_textbox(slide, Inches(0.7), Inches(4.25), Inches(5.4), Inches(0.5),
            "零样本异常检测路线（WinCLIP等）", font_size=17, bold=True, color=COLOR_GREEN)
add_textbox(slide, Inches(0.75), Inches(4.8), Inches(5.3), Inches(1.4),
            "仅需正常样本建库，适配小批量换产场景，但缺少深度分析能力",
            font_size=15, color=COLOR_GRAY)

# 大模型RAG应用
add_textbox(slide, Inches(6.8), Inches(1.1), Inches(6), Inches(0.5),
            "大模型与RAG的工业应用", font_size=20, bold=True, color=COLOR_PRIMARY)
add_rounded_rect(slide, Inches(6.8), Inches(1.7), Inches(6.0), Inches(2.2),
                 RGBColor(0xF3,0xE5,0xF5), RGBColor(0x6A,0x1B,0x9A))
add_textbox(slide, Inches(7.0), Inches(1.85), Inches(5.6), Inches(0.5),
            "生成式VLM + RAG知识库", font_size=17, bold=True, color=RGBColor(0x6A,0x1B,0x9A))
add_textbox(slide, Inches(7.05), Inches(2.4), Inches(5.5), Inches(1.4),
            "开始应用于文档问答、故障分析，但多为纯文本方案，缺少与视觉检测的深度融合；幻觉不可控，不符合合规要求",
            font_size=15, color=COLOR_GRAY)

# 现有方案不足
add_rounded_rect(slide, Inches(6.8), Inches(4.1), Inches(6.0), Inches(2.2),
                 RGBColor(0xFF,0xEB,0xEE), COLOR_RED)
add_textbox(slide, Inches(7.0), Inches(4.25), Inches(5.6), Inches(0.5),
            "现有方案共性不足", font_size=17, bold=True, color=COLOR_RED)
add_textbox(slide, Inches(7.05), Inches(4.8), Inches(5.5), Inches(1.4),
            "要么只有检测没有分析，要么只有分析没有定位；缺少兼顾效率、精度、合规性的完整工作流方案",
            font_size=15, color=COLOR_GRAY)

# ============================================================
# 第6页：双链路门控核心机制（含实验图）
# ============================================================
slide = add_blank_slide()
add_page_header(slide, "三、系统架构 · 双链路门控核心机制", 6)

# 左侧：机制说明
add_textbox(slide, Inches(0.5), Inches(1.1), Inches(5.5), Inches(0.5),
            "置信度门控双链路", font_size=20, bold=True, color=COLOR_PRIMARY)

mechanisms = [
    ("感知层统一计算", "WinCLIP计算像素级异常分数，生成全局异常置信度"),
    ("高置信 → 快链路", "异常分数≥阈值，直接输出缺陷位置、异常等级、热力图，不触发VLM"),
    ("低置信 → 慢链路", "异常分数<阈值，裁剪缺陷→VLM描述→RAG匹配标准→完整分析报告"),
    ("三级反馈反哺", "人工结果入案例库→辅助调整阈值→标准库人工审核更新"),
]
for i, (title, desc) in enumerate(mechanisms):
    top = Inches(1.7 + i * 1.25)
    add_rounded_rect(slide, Inches(0.5), top, Inches(5.5), Inches(1.1),
                     COLOR_LIGHT_GRAY, COLOR_PRIMARY)
    add_textbox(slide, Inches(0.65), top + Inches(0.08), Inches(5.2), Inches(0.4),
                title, font_size=15, bold=True, color=COLOR_DARK)
    add_textbox(slide, Inches(0.65), top + Inches(0.5), Inches(5.2), Inches(0.55),
                desc, font_size=12, color=COLOR_GRAY)

# 阈值说明
add_textbox(slide, Inches(0.5), Inches(6.8), Inches(5.5), Inches(0.4),
            "阈值：中位数+1.5×MAD = 0.2243（MVTec metal_nut正常样本）",
            font_size=12, bold=True, color=COLOR_ACCENT)

# 右侧：双链路演示图
add_image_safe(slide, os.path.join(RESULT_DIR, "gating", "dual_path_demo.png"),
               Inches(6.3), Inches(1.2), width=Inches(6.7))

# ============================================================
# 第7页：四层一底座架构
# ============================================================
slide = add_blank_slide()
add_page_header(slide, "三、系统架构 · 四层一底座", 7)

layers = [
    ("应用交互层", "单图检测 | 批量处理 | 报告导出 | 知识库管理 | 历史记录", RGBColor(0x6A,0x1B,0x9A)),
    ("认知分析层", "VLM缺陷描述 | RAG标准检索 | 根因推理 | 结构化报告生成", COLOR_GREEN),
    ("门控路由层", "置信度计算 | 阈值判断 | 链路调度 | 结果冲突处理", COLOR_ACCENT),
    ("感知检测层", "WinCLIP正常特征库 | 多尺度滑窗检测 | 异常热力图 | 缺陷裁剪", COLOR_PRIMARY),
    ("数据输入层", "正常样本图 | 待检工件图 | 质检标准文档 | 历史案例文档", COLOR_GRAY),
]
for i, (name, content, color) in enumerate(layers):
    top = Inches(1.2 + i * 1.05)
    add_rounded_rect(slide, Inches(1.5), top, Inches(10.3), Inches(0.9),
                     COLOR_WHITE, color)
    add_rect(slide, Inches(1.5), top, Inches(2.2), Inches(0.9), color)
    add_textbox(slide, Inches(1.6), top + Inches(0.25), Inches(2.0), Inches(0.4),
                name, font_size=17, bold=True, color=COLOR_WHITE, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(3.9), top + Inches(0.28), Inches(7.7), Inches(0.4),
                content, font_size=14, color=COLOR_DARK)

# 底座
add_rounded_rect(slide, Inches(1.5), Inches(6.5), Inches(10.3), Inches(0.6),
                 RGBColor(0x37,0x47,0x4F), RGBColor(0x37,0x47,0x4F))
add_textbox(slide, Inches(1.6), Inches(6.6), Inches(10.1), Inches(0.4),
            "知识库底座：正常特征库 | 质检标准库 | 历史案例库 | 报告模板库",
            font_size=14, bold=True, color=COLOR_WHITE, alignment=PP_ALIGN.CENTER)

# ============================================================
# 第8页：页码级可溯源校验（含实验图）
# ============================================================
slide = add_blank_slide()
add_page_header(slide, "三、核心机制 · 页码级可溯源校验", 8)

# 左侧：机制说明
add_textbox(slide, Inches(0.5), Inches(1.1), Inches(5.5), Inches(0.5),
            "引用ID可溯源校验机制", font_size=20, bold=True, color=COLOR_PRIMARY)

steps = [
    ("① 片段ID化", "每个标准片段分配唯一引用ID，绑定原文、页码、标准号"),
    ("② RAG检索", "返回结果同时包含片段内容+引用ID+页码+标准号"),
    ("③ 生成约束", "Prompt强制VLM输出必须携带引用ID，不允许无引用结论"),
    ("④ 后处理校验", "自动检查输出ID是否在检索结果中，不通过自动标红"),
    ("⑤ 降级模式", "VLM输出异常时，降级为展示原始标准片段供人工选择"),
]
for i, (title, desc) in enumerate(steps):
    top = Inches(1.7 + i * 1.0)
    add_rounded_rect(slide, Inches(0.5), top, Inches(5.5), Inches(0.85),
                     COLOR_LIGHT_GRAY, COLOR_GREEN)
    add_textbox(slide, Inches(0.65), top + Inches(0.06), Inches(5.2), Inches(0.35),
                title, font_size=14, bold=True, color=COLOR_GREEN)
    add_textbox(slide, Inches(0.65), top + Inches(0.42), Inches(5.2), Inches(0.4),
                desc, font_size=11, color=COLOR_GRAY)

add_textbox(slide, Inches(0.5), Inches(6.85), Inches(5.5), Inches(0.4),
            "设计原则：不追求零幻觉，追求所有引用可校验、错误可识别",
            font_size=12, bold=True, color=COLOR_ACCENT)

# 右侧：校验通过/失败对比图
add_textbox(slide, Inches(6.3), Inches(1.1), Inches(6.5), Inches(0.4),
            "校验通过 vs 校验失败（编造引用）", font_size=16, bold=True, color=COLOR_DARK)
add_image_safe(slide, os.path.join(RESULT_DIR, "cognition", "ref_validation_pass.png"),
               Inches(6.3), Inches(1.6), width=Inches(6.7))
add_image_safe(slide, os.path.join(RESULT_DIR, "cognition", "ref_validation_fail.png"),
               Inches(6.3), Inches(4.5), width=Inches(6.7))

# ============================================================
# 第9页：创新点
# ============================================================
slide = add_blank_slide()
add_page_header(slide, "四、项目创新点", 9)

innovations = [
    ("应用架构创新", "置信度门控的双链路质检架构",
     "高置信快检、低置信精析，兼顾检测效率与分析深度；避免全量图都走大模型分析的算力浪费",
     COLOR_PRIMARY),
    ("场景工作流创新", "少样本全流程质检闭环",
     "覆盖建库→检测→定位→分析→报告→迭代完整流程；仅需10-20张正常样本，换产适配0.5-1天",
     COLOR_GREEN),
    ("工程实现创新", "引用ID绑定的可溯源校验机制",
     "所有判定依据绑定标准原文与页码，输出自动校验引用一致性；校验失败自动降级，符合工业合规要求",
     COLOR_ACCENT),
]
for i, (cat, title, desc, color) in enumerate(innovations):
    top = Inches(1.3 + i * 1.9)
    add_rounded_rect(slide, Inches(0.8), top, Inches(11.7), Inches(1.7),
                     COLOR_WHITE, color)
    add_rect(slide, Inches(0.8), top, Inches(0.15), Inches(1.7), color)
    add_textbox(slide, Inches(1.2), top + Inches(0.15), Inches(3), Inches(0.4),
                cat, font_size=14, bold=True, color=color)
    add_textbox(slide, Inches(1.2), top + Inches(0.55), Inches(10.5), Inches(0.5),
                title, font_size=22, bold=True, color=COLOR_DARK)
    add_textbox(slide, Inches(1.2), top + Inches(1.1), Inches(10.5), Inches(0.5),
                desc, font_size=14, color=COLOR_GRAY)

# 技术边界说明
add_rounded_rect(slide, Inches(0.8), Inches(7.0), Inches(11.7), Inches(0.4),
                 RGBColor(0xFF,0xF8,0xE1), COLOR_ACCENT)
add_textbox(slide, Inches(1.0), Inches(7.03), Inches(11.3), Inches(0.35),
            "技术边界：底层视觉模型、语言模型、RAG框架采用开源成熟组件，创新集中在应用架构、工作流设计与工程落地，不做模型层原创",
            font_size=12, bold=True, color=COLOR_ACCENT)

# ============================================================
# 第10页：技术与数据可行性
# ============================================================
slide = add_blank_slide()
add_page_header(slide, "五、可行性分析 · 技术与数据", 10)

# 技术可行性
add_textbox(slide, Inches(0.5), Inches(1.1), Inches(6), Inches(0.5),
            "技术可行性", font_size=20, bold=True, color=COLOR_PRIMARY)
tech_items = [
    ("感知层", "WinCLIP为工业异常检测公认开源基线，效果有公开数据集验证"),
    ("认知层", "Qwen-VL、LangChain、Chroma均为成熟开源组件，工程路径清晰"),
    ("参考基线", "已有同方向架构验证技术路线可行"),
]
for i, (name, desc) in enumerate(tech_items):
    top = Inches(1.7 + i * 1.3)
    add_rounded_rect(slide, Inches(0.5), top, Inches(5.8), Inches(1.1),
                     COLOR_LIGHT_GRAY, COLOR_PRIMARY)
    add_textbox(slide, Inches(0.7), top + Inches(0.1), Inches(5.4), Inches(0.4),
                name, font_size=16, bold=True, color=COLOR_DARK)
    add_textbox(slide, Inches(0.7), top + Inches(0.55), Inches(5.4), Inches(0.5),
                desc, font_size=13, color=COLOR_GRAY)

# 数据可行性
add_textbox(slide, Inches(6.8), Inches(1.1), Inches(6), Inches(0.5),
            "数据可行性", font_size=20, bold=True, color=COLOR_PRIMARY)
data_items = [
    ("图像数据", "MVTec-AD金属表面子集，原生包含正常样本与各类缺陷样本，匹配异常检测范式"),
    ("文档数据", "GB/T 34528-2017《金属材料 表面缺陷目视检测导则》等公开国家标准"),
    ("无需私有数据", "基于公开资料即可完成系统原型，不依赖企业私有数据"),
]
for i, (name, desc) in enumerate(data_items):
    top = Inches(1.7 + i * 1.3)
    add_rounded_rect(slide, Inches(6.8), top, Inches(6.0), Inches(1.1),
                     RGBColor(0xE8,0xF5,0xE9), COLOR_GREEN)
    add_textbox(slide, Inches(7.0), top + Inches(0.1), Inches(5.6), Inches(0.4),
                name, font_size=16, bold=True, color=COLOR_GREEN)
    add_textbox(slide, Inches(7.0), top + Inches(0.55), Inches(5.6), Inches(0.5),
                desc, font_size=13, color=COLOR_GRAY)

# 检测效果数据（来自实验）
add_rounded_rect(slide, Inches(0.5), Inches(5.8), Inches(12.3), Inches(1.3),
                 RGBColor(0xE3,0xF2,0xFD), COLOR_PRIMARY)
add_textbox(slide, Inches(0.7), Inches(5.9), Inches(12), Inches(0.4),
            "先导实验检测效果（MVTec-AD metal_nut，零样本无训练）",
            font_size=16, bold=True, color=COLOR_PRIMARY)
metrics_text = "AUROC = 0.9868    |    准确率 = 94.78%    |    精确率 = 96.77%    |    召回率 = 96.77%    |    阈值 = 0.2243"
add_textbox(slide, Inches(0.7), Inches(6.4), Inches(12), Inches(0.5),
            metrics_text, font_size=18, bold=True, color=COLOR_DARK, alignment=PP_ALIGN.CENTER)

# ============================================================
# 第11页：算力与周期可行性
# ============================================================
slide = add_blank_slide()
add_page_header(slide, "五、可行性分析 · 算力与周期", 11)

# 算力
add_textbox(slide, Inches(0.5), Inches(1.1), Inches(12), Inches(0.5),
            "算力可行性", font_size=20, bold=True, color=COLOR_PRIMARY)

compute_items = [
    ("感知检测流程", "纯CPU（i7级）单图2-3秒，可实时输出热力图", COLOR_PRIMARY),
    ("认知分析流程", "纯CPU单图约20-30秒（异步离线处理可接受）；GPU环境下5-8秒", COLOR_GREEN),
    ("演示方案", "模型预加载 + 固定样例兜底，保证现场演示流畅", COLOR_ACCENT),
]
for i, (name, desc, color) in enumerate(compute_items):
    left = Inches(0.5 + i * 4.2)
    add_rounded_rect(slide, left, Inches(1.7), Inches(3.9), Inches(1.8),
                     COLOR_WHITE, color)
    add_rect(slide, left, Inches(1.7), Inches(3.9), Inches(0.55), color)
    add_textbox(slide, left + Inches(0.15), Inches(1.78), Inches(3.6), Inches(0.4),
                name, font_size=16, bold=True, color=COLOR_WHITE)
    add_textbox(slide, left + Inches(0.2), Inches(2.4), Inches(3.5), Inches(1.0),
                desc, font_size=13, color=COLOR_GRAY)

# 周期
add_textbox(slide, Inches(0.5), Inches(3.9), Inches(12), Inches(0.5),
            "周期可行性", font_size=20, bold=True, color=COLOR_PRIMARY)

add_rounded_rect(slide, Inches(0.5), Inches(4.5), Inches(12.3), Inches(2.5),
                 COLOR_LIGHT_GRAY, COLOR_PRIMARY)
add_textbox(slide, Inches(0.8), Inches(4.65), Inches(11.7), Inches(0.4),
            "无模型训练任务，以工程组装和场景适配为主，12周分三阶段交付",
            font_size=16, bold=True, color=COLOR_DARK)

phases = [
    ("开题阶段", "第4周", "最小可行性Demo、开题PPT、书面报告", "15%"),
    ("中期阶段", "第9-10周", "核心功能完整Demo、中期报告、指标初测", "25%"),
    ("结题阶段", "第16周", "完整系统、结题报告、演示视频", "60%"),
]
for i, (name, time, deliver, ratio) in enumerate(phases):
    left = Inches(0.8 + i * 4.0)
    add_rounded_rect(slide, left, Inches(5.3), Inches(3.7), Inches(1.5),
                     COLOR_WHITE, COLOR_PRIMARY)
    add_textbox(slide, left + Inches(0.15), Inches(5.4), Inches(3.4), Inches(0.35),
                f"{name}（{time}）", font_size=14, bold=True, color=COLOR_PRIMARY)
    add_textbox(slide, left + Inches(0.15), Inches(5.8), Inches(3.4), Inches(0.6),
                deliver, font_size=11, color=COLOR_GRAY)
    add_textbox(slide, left + Inches(0.15), Inches(6.4), Inches(3.4), Inches(0.35),
                f"考核占比：{ratio}", font_size=13, bold=True, color=COLOR_ACCENT)

# ============================================================
# 第12页：先导实验（含实验图）
# ============================================================
slide = add_blank_slide()
add_page_header(slide, "五、可行性分析 · 先导实验", 12)

add_textbox(slide, Inches(0.5), Inches(1.05), Inches(12), Inches(0.4),
            "基于MVTec-AD metal_nut公开数据集，验证双链路门控架构与可溯源机制可行性",
            font_size=15, bold=True, color=COLOR_ACCENT)

# 上排：感知层结果
add_textbox(slide, Inches(0.5), Inches(1.55), Inches(6), Inches(0.35),
            "感知层：零样本异常检测", font_size=15, bold=True, color=COLOR_PRIMARY)
add_image_safe(slide, os.path.join(RESULT_DIR, "perception", "score_distribution.png"),
               Inches(0.3), Inches(1.95), width=Inches(4.2))
add_image_safe(slide, os.path.join(RESULT_DIR, "perception", "high_conf_sample.png"),
               Inches(4.4), Inches(1.95), width=Inches(2.5))

# 上排右：认知层结果
add_textbox(slide, Inches(7.2), Inches(1.55), Inches(6), Inches(0.35),
            "认知层：RAG检索 + 引用校验", font_size=15, bold=True, color=COLOR_GREEN)
add_image_safe(slide, os.path.join(RESULT_DIR, "cognition", "retrieval_result.png"),
               Inches(7.0), Inches(1.95), width=Inches(5.8))

# 下排：门控 + 低置信
add_textbox(slide, Inches(0.5), Inches(4.6), Inches(6), Inches(0.35),
            "门控层：高/低置信双链路对比", font_size=15, bold=True, color=COLOR_ACCENT)
add_image_safe(slide, os.path.join(RESULT_DIR, "perception", "low_conf_sample.png"),
               Inches(0.3), Inches(5.0), width=Inches(2.5))
add_image_safe(slide, os.path.join(RESULT_DIR, "cognition", "cognition_demo.png"),
               Inches(2.9), Inches(5.0), width=Inches(4.0))

# 右下：实验结论
add_rounded_rect(slide, Inches(7.2), Inches(4.6), Inches(5.8), Inches(2.5),
                 RGBColor(0xE8,0xF5,0xE9), COLOR_GREEN)
add_textbox(slide, Inches(7.4), Inches(4.7), Inches(5.4), Inches(0.4),
            "先导实验结论", font_size=16, bold=True, color=COLOR_GREEN)
conclusions = [
    "感知层：零样本AUROC=0.9868，检测效果优异",
    "门控层：高/低置信样例可正确路由，机制跑通",
    "认知层：RAG检索准确，引用ID校验通过/失败可正确识别",
    "核心技术路线可行性已验证，可进入中期开发",
]
for i, c in enumerate(conclusions):
    add_textbox(slide, Inches(7.5), Inches(5.15 + i * 0.45), Inches(5.3), Inches(0.4),
                f"• {c}", font_size=12, color=COLOR_DARK)

# ============================================================
# 第13页：评估指标与对比实验
# ============================================================
slide = add_blank_slide()
add_page_header(slide, "六、评估指标与对比实验", 13)

# 三层指标
add_textbox(slide, Inches(0.5), Inches(1.1), Inches(6), Inches(0.5),
            "三层量化评估体系", font_size=20, bold=True, color=COLOR_PRIMARY)

metric_layers = [
    ("检测层", "AUROC、最佳F1、定位IoU、误报率、漏报率", COLOR_PRIMARY),
    ("认知层", "标准匹配Top-1准确率、引用准确率、缺陷描述召回率、幻觉率", COLOR_GREEN),
    ("系统层", "单图平均处理耗时、报告生成耗时、人工操作步骤减少率", COLOR_ACCENT),
]
for i, (name, metrics, color) in enumerate(metric_layers):
    top = Inches(1.7 + i * 1.3)
    add_rounded_rect(slide, Inches(0.5), top, Inches(5.8), Inches(1.1),
                     COLOR_WHITE, color)
    add_rect(slide, Inches(0.5), top, Inches(1.5), Inches(1.1), color)
    add_textbox(slide, Inches(0.55), top + Inches(0.35), Inches(1.4), Inches(0.4),
                name, font_size=16, bold=True, color=COLOR_WHITE, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(2.2), top + Inches(0.3), Inches(4.0), Inches(0.6),
                metrics, font_size=12, color=COLOR_DARK)

# 对比实验
add_textbox(slide, Inches(6.8), Inches(1.1), Inches(6), Inches(0.5),
            "分阶段对比实验设计", font_size=20, bold=True, color=COLOR_PRIMARY)

comparisons = [
    ("中期", "纯WinCLIP vs 双链路", "验证双链路的分析价值", COLOR_PRIMARY),
    ("中期", "纯VLM vs 双链路", "验证感知前置的精度提升", COLOR_GREEN),
    ("结题", "无门控串行 vs 双链路", "验证门控的效率价值", COLOR_ACCENT),
]
for i, (stage, compare, purpose, color) in enumerate(comparisons):
    top = Inches(1.7 + i * 1.3)
    add_rounded_rect(slide, Inches(6.8), top, Inches(6.0), Inches(1.1),
                     COLOR_LIGHT_GRAY, color)
    add_textbox(slide, Inches(7.0), top + Inches(0.1), Inches(1.2), Inches(0.4),
                stage, font_size=14, bold=True, color=color)
    add_textbox(slide, Inches(8.3), top + Inches(0.1), Inches(4.3), Inches(0.4),
                compare, font_size=14, bold=True, color=COLOR_DARK)
    add_textbox(slide, Inches(7.0), top + Inches(0.55), Inches(5.6), Inches(0.4),
                f"目的：{purpose}", font_size=12, color=COLOR_GRAY)

# 对比维度说明
add_rounded_rect(slide, Inches(0.5), Inches(5.8), Inches(12.3), Inches(1.3),
                 RGBColor(0xFF,0xF8,0xE1), COLOR_ACCENT)
add_textbox(slide, Inches(0.7), Inches(5.9), Inches(12), Inches(0.4),
            "对比维度说明", font_size=15, bold=True, color=COLOR_ACCENT)
add_textbox(slide, Inches(0.7), Inches(6.3), Inches(12), Inches(0.7),
            "纯WinCLIP比检测层+系统层；纯VLM比认知层+系统层；无门控串行比系统层平均耗时。不同方案定位不同，分维度对比，各自比对应的优势",
            font_size=13, color=COLOR_DARK)

# ============================================================
# 第14页：进度计划与团队分工
# ============================================================
slide = add_blank_slide()
add_page_header(slide, "七、进度计划与团队分工", 14)

# 进度计划
add_textbox(slide, Inches(0.5), Inches(1.1), Inches(7), Inches(0.4),
            "项目进度计划（严格对应课程节点）", font_size=18, bold=True, color=COLOR_PRIMARY)

schedule = [
    ("第1-3周", "选题调研 + 先导实验", "环境搭建、WinCLIP复现、门控验证、认知层验证"),
    ("第4周", "开题答辩", "最小Demo、开题PPT、书面报告（15%）"),
    ("第5-8周", "核心开发", "感知层完善、认知层实现、门控路由、界面原型"),
    ("第9-10周", "中期检查", "核心功能Demo、中期报告、对比实验初测（25%）"),
    ("第11-15周", "系统集成", "批量处理、报告导出、反馈机制、完整对比实验"),
    ("第16周", "结题答辩", "完整系统、结题报告、演示视频（60%）"),
]
for i, (time, task, detail) in enumerate(schedule):
    top = Inches(1.6 + i * 0.85)
    add_rounded_rect(slide, Inches(0.5), top, Inches(7.0), Inches(0.75),
                     COLOR_LIGHT_GRAY if i % 2 == 0 else COLOR_WHITE, COLOR_PRIMARY)
    add_textbox(slide, Inches(0.6), top + Inches(0.08), Inches(1.3), Inches(0.3),
                time, font_size=12, bold=True, color=COLOR_PRIMARY)
    add_textbox(slide, Inches(2.0), top + Inches(0.08), Inches(2.0), Inches(0.3),
                task, font_size=12, bold=True, color=COLOR_DARK)
    add_textbox(slide, Inches(2.0), top + Inches(0.4), Inches(5.3), Inches(0.3),
                detail, font_size=10, color=COLOR_GRAY)

# 团队分工
add_textbox(slide, Inches(8.0), Inches(1.1), Inches(5), Inches(0.4),
            "团队分工（3人）", font_size=18, bold=True, color=COLOR_PRIMARY)

roles = [
    ("项目负责人", "总架构、认知层、文档与答辩主讲", COLOR_PRIMARY),
    ("感知算法", "WinCLIP部署、多尺度优化、指标测试", COLOR_GREEN),
    ("系统支持", "界面开发、文档排版、演示保障", COLOR_ACCENT),
]
for i, (role, duty, color) in enumerate(roles):
    top = Inches(1.6 + i * 1.5)
    add_rounded_rect(slide, Inches(8.0), top, Inches(4.8), Inches(1.3),
                     COLOR_WHITE, color)
    add_rect(slide, Inches(8.0), top, Inches(0.12), Inches(1.3), color)
    add_textbox(slide, Inches(8.3), top + Inches(0.15), Inches(4.3), Inches(0.4),
                role, font_size=16, bold=True, color=color)
    add_textbox(slide, Inches(8.3), top + Inches(0.6), Inches(4.3), Inches(0.6),
                duty, font_size=13, color=COLOR_GRAY)

# 同步规则
add_rounded_rect(slide, Inches(8.0), Inches(6.2), Inches(4.8), Inches(0.9),
                 RGBColor(0xE3,0xF2,0xFD), COLOR_PRIMARY)
add_textbox(slide, Inches(8.2), Inches(6.3), Inches(4.4), Inches(0.35),
            "协作规则", font_size=13, bold=True, color=COLOR_PRIMARY)
add_textbox(slide, Inches(8.2), Inches(6.65), Inches(4.4), Inches(0.4),
            "每日进度同步 · 接口前置 · 先有后优 · 风险早报",
            font_size=11, color=COLOR_DARK)

# ============================================================
# 第15页：风险预案与总结
# ============================================================
slide = add_blank_slide()
add_page_header(slide, "七、风险预案与总结展望", 15)

# 风险预案
add_textbox(slide, Inches(0.5), Inches(1.1), Inches(12), Inches(0.4),
            "风险预案", font_size=20, bold=True, color=COLOR_PRIMARY)

risks = [
    ("WinCLIP效果不达预期", "替换PatchCore/Anomalib基线，接口兼容"),
    ("VLM速度过慢", "切换轻量化模型，或高频场景模板化生成"),
    ("RAG匹配不准/引用错误", "优化分块策略，人工预置核心标准条目"),
    ("门控阈值不合理", "中位数+1.5×MAD辅助标定，支持人工微调"),
    ("JSON Schema不服从", "正则提取中间层；成功率低直接降级人工选"),
    ("真实样本获取困难", "公开真实工业数据集+模拟场景干扰做跨域验证"),
]
for i, (risk, solution) in enumerate(risks):
    row = i // 2
    col = i % 2
    left = Inches(0.5 + col * 6.2)
    top = Inches(1.6 + row * 1.1)
    add_rounded_rect(slide, left, top, Inches(5.9), Inches(0.95),
                     RGBColor(0xFF,0xEB,0xEE), COLOR_RED)
    add_textbox(slide, left + Inches(0.15), top + Inches(0.08), Inches(5.6), Inches(0.35),
                f"风险：{risk}", font_size=13, bold=True, color=COLOR_RED)
    add_textbox(slide, left + Inches(0.15), top + Inches(0.48), Inches(5.6), Inches(0.4),
                f"应对：{solution}", font_size=12, color=COLOR_DARK)

# 总结
add_rounded_rect(slide, Inches(0.5), Inches(5.1), Inches(12.3), Inches(1.5),
                 RGBColor(0xE3,0xF2,0xFD), COLOR_PRIMARY)
add_textbox(slide, Inches(0.8), Inches(5.25), Inches(11.7), Inches(0.4),
            "项目总结", font_size=18, bold=True, color=COLOR_PRIMARY)
add_textbox(slide, Inches(0.8), Inches(5.7), Inches(11.7), Inches(0.8),
            "针对金属表面质检少样本换产难、分析效率低、合规性弱的痛点，提出置信度门控的双链路架构，"
            "实现可溯源的全流程质检辅助。先导实验已验证核心技术路线可行性（AUROC=0.9868），符合应用实践定位。",
            font_size=14, color=COLOR_DARK)

# 致谢
add_textbox(slide, Inches(0.5), Inches(6.8), Inches(12.3), Inches(0.5),
            "感谢各位老师聆听，恳请批评指正！", font_size=22, bold=True,
            color=COLOR_DARK, alignment=PP_ALIGN.CENTER)

# ============================================================
# 保存
# ============================================================
prs.save(OUT_PATH)
print(f"PPT已生成: {OUT_PATH}")
print(f"总页数: {len(prs.slides)}")
