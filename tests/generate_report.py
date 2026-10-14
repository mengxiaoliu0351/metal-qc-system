# -*- coding: utf-8 -*-
"""
开题书面报告生成脚本
"""
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

ROOT = r"D:\programme file\ResearchArticle"
RESULT_DIR = os.path.join(ROOT, "results")
OUT_PATH = os.path.join(ROOT, "开题报告_双链路智能质检辅助系统.docx")

doc = Document()

# 设置默认字体
style = doc.styles['Normal']
font = style.font
font.name = '宋体'
font.size = Pt(12)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = '黑体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        run.font.color.rgb = RGBColor(0x15, 0x65, 0xC0)
    return h

def add_para(text, bold=False, size=12, align=None, indent=True):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Pt(24)
    p.paragraph_format.line_spacing = 1.5
    run = p.add_run(text)
    run.font.name = '宋体'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    run.font.size = Pt(size)
    run.font.bold = bold
    if align:
        p.alignment = align
    return p

def add_bullet(text, level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.left_indent = Pt(24 + level * 24)
    run = p.add_run(text)
    run.font.name = '宋体'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    run.font.size = Pt(12)
    return p

def add_table(headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # 表头
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.bold = True
                run.font.size = Pt(11)
                run.font.name = '宋体'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    # 数据行
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.rows[r + 1].cells[c]
            cell.text = str(val)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10.5)
                    run.font.name = '宋体'
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    doc.add_paragraph()
    return table

def add_image_safe(img_path, width=Inches(5.5)):
    if os.path.exists(img_path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(img_path, width=width)
        return True
    return False

# ============================================================
# 封面
# ============================================================
for _ in range(4):
    doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("《人工智能应用实践》课程")
run.font.size = Pt(22)
run.font.name = '黑体'
run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("开题报告")
run.font.size = Pt(36)
run.font.bold = True
run.font.name = '黑体'
run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("面向金属表面缺陷的双链路门控智能质检辅助系统")
run.font.size = Pt(20)
run.font.bold = True
run.font.name = '黑体'
run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

for _ in range(6):
    doc.add_paragraph()

info_lines = [
    "学院：山东大学人工智能学院",
    "团队成员：XXX / XXX / XXX",
    "指导教师：赵玉琨 副教授",
    "时间：2026年9月",
]
for line in info_lines:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(line)
    run.font.size = Pt(14)
    run.font.name = '宋体'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

doc.add_page_break()

# ============================================================
# 一、项目背景与意义
# ============================================================
add_heading("一、项目背景与意义", 1)

add_heading("1.1 研究背景", 2)
add_para("制造业是国民经济的支柱产业，产品质量控制是制造业的核心环节。"
         "在金属零件加工领域，表面缺陷检测是质量控制的关键工序，直接影响产品的使用寿命和安全性。"
         "当前，中小机械加工厂普遍面临多品种、小批量的生产模式，产线换产频繁，"
         "传统的质检方式难以适应这种快速变化的生产需求。")

add_para("现有的工业缺陷检测方案主要存在以下三个方面的痛点：")
add_bullet("换产成本高：基于监督学习的缺陷检测模型需要大量标注数据进行训练，新零件换产时重新标注和训练的周期长、成本高，小批量产线难以承担。")
add_bullet("分析效率低：质检过程中查阅标准、撰写报告、分析根因高度依赖人工经验，单份质检报告耗时30分钟以上，效率低下且容易出错。")
add_bullet("合规性弱：人工判定标准不统一，检测结论缺乏可溯源记录，不符合工业质检的可追溯性要求，难以满足质量体系认证的需要。")

add_heading("1.2 研究意义", 2)
add_para("本项目旨在开发一套面向金属表面缺陷的双链路门控智能质检辅助系统，"
         "通过融合零样本异常检测技术与大模型检索增强生成技术，实现少样本条件下的缺陷检测、"
         "标准匹配、分析报告生成的全流程辅助。项目的研究意义体现在以下几个方面：")
add_bullet("理论意义：探索置信度门控的双链路架构在工业质检场景中的应用，为视觉检测与大模型分析的融合提供新的架构思路。")
add_bullet("应用价值：解决中小制造企业多品种小批量产线质检换产难、分析效率低的实际痛点，降低质检成本，提升质检效率。")
add_bullet("工程实践：通过完整的系统开发与验证，培养团队的工程实践能力，积累工业AI应用的开发经验。")

# ============================================================
# 二、国内外研究现状
# ============================================================
add_heading("二、国内外研究现状", 1)

add_heading("2.1 工业缺陷检测技术路线", 2)
add_para("工业缺陷检测技术主要分为两条技术路线：")
add_bullet("监督学习路线：以YOLO、CNN等深度学习模型为代表，通过大量标注数据训练检测模型，具有检测精度高的优点。但该路线需要大量标注数据，模型训练周期长，换产时重新标注和训练的成本高，在少样本场景下适配性差。")
add_bullet("零样本异常检测路线：以WinCLIP等为代表，仅需正常样本构建特征记忆库，通过计算待检图像与正常特征的偏离度进行异常检测。该路线无需标注缺陷数据，适配小批量换产场景，但通常只具备检测定位能力，缺乏深度分析能力。")

add_heading("2.2 大模型与RAG在工业中的应用", 2)
add_para("近年来，生成式视觉语言模型（VLM）和检索增强生成（RAG）技术快速发展，"
         "开始在工业领域得到应用。VLM能够理解图像内容并生成自然语言描述，"
         "RAG技术能够结合领域知识库进行精准问答。然而，现有方案多为纯文本问答或单一图像分析，"
         "缺少与视觉检测能力的深度融合，且大模型存在幻觉问题，输出结论不可控，不符合工业质检的合规性要求。")

add_heading("2.3 现有方案的不足", 2)
add_para("综合分析现有方案，主要存在以下共性不足：")
add_bullet("功能单一：纯检测方案只输出缺陷位置和类别，没有分析和报告能力；纯大模型方案只有分析能力，缺乏精确定位能力。")
add_bullet("效率与精度难以兼顾：所有图像都走完整的大模型分析流程，算力消耗大、耗时长；而纯检测方案又缺乏分析深度。")
add_bullet("合规性不足：大模型输出结论缺乏标准溯源，存在幻觉风险，不符合工业质检的可追溯性要求。")
add_bullet("落地成本高：现有工业质检方案多绑定特定产线，定制化成本高，中小工厂难以快速复用。")

# ============================================================
# 三、研究内容与技术路线
# ============================================================
add_heading("三、研究内容与技术路线", 1)

add_heading("3.1 研究内容", 2)
add_para("本项目的主要研究内容包括以下几个方面：")
add_bullet("感知检测层研究：基于WinCLIP架构，研究零样本异常检测方法，实现多尺度滑窗检测、异常热力图生成、缺陷区域裁剪。")
add_bullet("认知分析层研究：基于VLM和RAG技术，研究缺陷描述生成、质检标准检索、根因推理、结构化报告生成方法。")
add_bullet("门控路由层研究：研究基于异常分数的置信度门控机制，实现高置信样本快链路检测、低置信样本慢链路深度分析的智能路由。")
add_bullet("可溯源校验机制研究：研究基于引用ID的标准溯源校验方法，实现所有判定结论的标准原文与页码绑定，以及输出一致性校验。")
add_bullet("系统集成与界面开发：开发Web交互界面，实现单图检测、批量处理、报告导出、知识库管理、历史记录等功能。")

add_heading("3.2 技术路线", 2)
add_para("本项目采用四层一底座的系统架构，技术路线如下：")
add_para("数据输入层：接入正常样本图像、待检工件图像、质检标准文档、历史案例文档，进行格式标准化预处理。")
add_para("感知检测层：采用WinCLIP架构，用正常样本构建特征记忆库，通过多尺度滑窗计算像素级异常分数，生成异常热力图，定位并裁剪缺陷区域。")
add_para("门控路由层：基于异常分数计算全局置信度，采用中位数+1.5×MAD确定门控阈值。高置信样本走快链路直接输出检测结果，低置信样本触发慢链路深度分析。")
add_para("认知分析层：对裁剪的缺陷区域，用VLM生成自然语言描述，用RAG检索匹配质检标准，进行根因推理，生成结构化分析报告。")
add_para("应用交互层：提供Web交互界面，支持单图检测、批量处理、报告导出、知识库管理、历史记录查询等功能。")
add_para("知识库底座：统一存储正常特征库、质检标准库、历史案例库、报告模板库，支持人工审核更新。")

# ============================================================
# 四、系统架构设计
# ============================================================
add_heading("四、系统架构设计", 1)

add_heading("4.1 双链路门控核心机制", 2)
add_para("本项目的核心创新机制是置信度门控的双链路架构，具体机制如下：")
add_bullet("感知层统一计算：输入图像经WinCLIP计算像素级异常分数，生成全局异常置信度。")
add_bullet("高置信快链路：异常分数大于等于阈值的样本，直接输出缺陷位置、异常等级、热力图，不触发VLM分析，保证检测效率。")
add_bullet("低置信慢链路：异常分数小于阈值的样本，裁剪缺陷区域后送入VLM生成描述，再经RAG匹配标准，输出完整分析报告，保证分析精度。")
add_bullet("三级反馈反哺：人工复核结果存入历史案例库；定期根据案例数据辅助调整异常分数阈值；标准库更新需人工审核后发布，避免知识库污染。")

add_para("门控阈值采用中位数+1.5×MAD（中位数绝对偏差）的方法确定，"
         "该方法在小样本下比分位数更稳健，能够有效避免极端值的影响。"
         "初始阈值基于正常样本的异常分数分布计算，实际部署时支持人工微调。")

add_heading("4.2 页码级可溯源校验机制", 2)
add_para("为解决大模型幻觉问题，满足工业质检的合规性要求，本项目设计了引用ID可溯源校验机制：")
add_bullet("片段ID化：知识库中每个标准片段分配唯一引用ID，绑定原文内容、页码、标准号。")
add_bullet("RAG检索：检索结果同时包含片段内容、引用ID、页码、标准号。")
add_bullet("生成约束：Prompt强制要求VLM输出结论必须携带引用ID，不允许无引用生成结论。")
add_bullet("后处理校验：自动检查输出中所有引用ID是否存在于本次检索结果中，校验通过正常显示，校验不通过自动标红提示人工复核。")
add_bullet("降级模式：VLM输出格式异常时，自动降级为展示RAG检索到的原始标准片段，由人工选择适用条款。")

add_para("该机制的设计原则是：不追求零幻觉，追求所有引用可校验、错误可识别，"
         "确保系统输出的所有判定结论都有标准依据可追溯，符合工业质检的合规性要求。")

# ============================================================
# 五、项目创新点
# ============================================================
add_heading("五、项目创新点", 1)

add_para("本项目的创新点集中在应用架构、场景工作流和工程实现三个层面，"
         "底层视觉模型、语言模型、RAG框架均采用开源成熟组件，不做模型层原创。")

add_heading("5.1 应用架构创新：置信度门控的双链路质检架构", 2)
add_para("提出置信度门控的双链路质检架构，高置信样本走快链路直接输出检测结果，"
         "低置信样本走慢链路进行深度分析。该架构兼顾了检测效率与分析深度，"
         "避免了所有图像都走大模型分析的算力浪费，同时保证了边缘案例的分析精度。"
         "与纯检测方案相比，增加了分析能力；与纯大模型方案相比，增加了精确定位能力。")

add_heading("5.2 场景工作流创新：少样本全流程质检闭环", 2)
add_para("设计覆盖「建库→检测→定位→分析→报告→迭代」的完整质检工作流，"
         "仅需10-20张正常样本即可构建特征库，无需标注缺陷数据，"
         "换产适配周期仅需0.5-1天，适配中小制造企业多品种小批量的生产场景。"
         "该工作流实现了从缺陷检测到报告生成的全流程自动化，显著降低了质检人工工作量。")

add_heading("5.3 工程实现创新：引用ID绑定的可溯源校验机制", 2)
add_para("设计基于引用ID的可溯源校验机制，所有判定依据绑定标准原文与页码，"
         "输出自动校验引用一致性，校验失败自动降级为人工选择。"
         "该机制有效解决了大模型幻觉问题，确保所有结论可溯源、错误可识别，"
         "符合工业质检的合规性要求，为大模型在工业合规场景中的应用提供了工程实践参考。")

# ============================================================
# 六、可行性分析
# ============================================================
add_heading("六、可行性分析", 1)

add_heading("6.1 技术可行性", 2)
add_para("感知层：WinCLIP是工业异常检测领域的公认开源基线，在MVTec-AD等公开数据集上有优异的检测效果，技术路线成熟可靠。")
add_para("认知层：Qwen-VL、LangChain、Chroma等组件均为成熟的开源项目，具有完善的文档和活跃的社区支持，工程落地路径清晰。")
add_para("参考基线：已有SemiFA、MMAD等同方向的研究工作验证了视觉检测与大模型分析融合的技术路线可行性。")

add_heading("6.2 数据可行性", 2)
add_para("图像数据：采用MVTec-AD工业异常检测数据集的金属表面子集（metal_nut、screw），"
         "该数据集原生包含正常样本与各类缺陷样本，完全匹配零样本异常检测范式，可直接用于系统开发与测试。")
add_para("文档数据：采用GB/T 34528-2017《金属材料 表面缺陷目视检测导则》等公开国家标准，"
         "以及通用质检报告模板，可构建质检标准知识库。")
add_para("本项目无需企业私有数据，基于公开资料即可完成系统原型开发与验证。")

add_heading("6.3 算力可行性", 2)
add_para("感知检测流程：纯CPU（i7级）环境下单图处理时间约2-3秒，可实时输出热力图和检测结果。")
add_para("认知分析流程：纯CPU环境下单图处理时间约20-30秒，采用异步离线处理模式可满足使用需求；"
         "有GPU环境时可加速至5-8秒。")
add_para("演示方案：采用模型预加载+固定样例兜底的方式，保证现场演示流畅。")

add_heading("6.4 先导实验验证", 2)
add_para("为验证核心技术路线的可行性，项目组已完成以下先导实验：")

add_para("（1）感知层零样本异常检测实验", bold=True, indent=False)
add_para("基于MVTec-AD metal_nut数据集，采用CLIP ViT-B-16模型构建正常样本特征记忆库，"
         "对115张测试图像（含93张缺陷样本、22张正常样本）进行零样本异常检测。"
         "实验结果表明，零样本无训练条件下检测AUROC达到0.9868，"
         "在阈值0.2243下准确率达到94.78%，精确率96.77%，召回率96.77%，检测效果优异。")

add_image_safe(os.path.join(RESULT_DIR, "perception", "score_distribution.png"), width=Inches(5.0))

add_para("（2）门控机制验证实验", bold=True, indent=False)
add_para("选取高置信样本（异常分数远高于阈值）和低置信样本（异常分数接近阈值）各一张，"
         "验证双链路门控路由机制。实验结果表明，高置信样本可正确路由到快链路直接输出检测结果，"
         "低置信样本可正确触发慢链路进行深度分析，门控机制运行正常。")

add_image_safe(os.path.join(RESULT_DIR, "gating", "dual_path_demo.png"), width=Inches(5.5))

add_para("（3）认知层RAG检索与引用校验实验", bold=True, indent=False)
add_para("构建包含10条金属表面缺陷质检标准片段的知识库，每条片段分配唯一引用ID。"
         "采用TF-IDF+余弦相似度进行RAG检索，验证标准检索的准确性和引用ID校验机制。"
         "实验结果表明，RAG检索能够准确匹配相关标准条款，引用ID校验能够正确识别通过和失败（编造引用）的情况，"
         "校验失败时可自动降级为展示原始标准片段供人工选择。")

add_image_safe(os.path.join(RESULT_DIR, "cognition", "cognition_demo.png"), width=Inches(5.5))

add_para("先导实验结论：核心技术路线可行性已验证，感知层检测效果优异，门控机制和可溯源校验机制运行正常，"
         "可进入中期系统开发阶段。")

# ============================================================
# 七、评估指标与对比实验
# ============================================================
add_heading("七、评估指标与对比实验", 1)

add_heading("7.1 三层量化评估体系", 2)
add_table(
    ["评估层级", "评估指标"],
    [
        ["检测层", "AUROC、最佳F1值、定位IoU、误报率、漏报率"],
        ["认知层", "标准匹配Top-1准确率、引用准确率、缺陷描述召回率、幻觉率"],
        ["系统层", "单图平均处理耗时、报告生成耗时、人工操作步骤减少率"],
    ]
)

add_heading("7.2 分阶段对比实验设计", 2)
add_table(
    ["阶段", "对比组别", "对比目的"],
    [
        ["中期", "纯WinCLIP检测 vs 双链路系统", "验证双链路的分析价值"],
        ["中期", "纯VLM整图分析 vs 双链路系统", "验证感知前置的精度提升"],
        ["结题", "无门控全串行 vs 双链路系统", "验证门控的效率价值"],
    ]
)
add_para("对比维度说明：纯WinCLIP组比检测层+系统层指标；纯VLM组比认知层+系统层指标；"
         "无门控串行组比系统层平均耗时。不同方案定位不同，分维度对比，各自比对应的优势。")

# ============================================================
# 八、进度计划
# ============================================================
add_heading("八、进度计划", 1)
add_table(
    ["阶段", "时间", "核心任务", "交付物", "考核占比"],
    [
        ["选题调研", "第1-2周", "场景确定、技术调研、环境搭建", "调研笔记、技术选型", "-"],
        ["先导实验", "第3周", "WinCLIP复现、门控验证、认知层验证", "实验截图、最小Demo", "-"],
        ["开题答辩", "第4周", "整理开题材料、答辩", "开题PPT、书面报告", "15%"],
        ["核心开发I", "第5-6周", "感知层完善、门控路由实现", "感知模块、门控路由", "-"],
        ["核心开发II", "第7-8周", "认知层实现、界面原型", "认知模块、界面原型", "-"],
        ["中期检查", "第9-10周", "双链路跑通、对比实验初测", "中期Demo、中期报告", "25%"],
        ["系统集成", "第11-13周", "批量处理、报告导出、反馈机制", "完整系统原型", "-"],
        ["结题准备", "第14-15周", "指标测试、结题报告、演示视频", "结题材料", "-"],
        ["结题答辩", "第16周", "最终答辩、系统演示", "结题PPT、完整系统", "60%"],
    ]
)

# ============================================================
# 九、团队分工
# ============================================================
add_heading("九、团队分工", 1)
add_table(
    ["角色", "核心职责", "关键交付物"],
    [
        ["项目负责人", "总架构设计、进度把控、答辩主讲；认知层核心逻辑、文档总纂", "完整PPT、开题报告、认知层验证、答辩问答库"],
        ["感知算法工程师", "WinCLIP部署、多尺度优化、阈值标定、指标测试", "感知模块、热力图、检测指标数据"],
        ["系统支持工程师", "界面开发、文档排版、演示保障", "Web界面、格式规范的报告、演示视频"],
    ]
)
add_para("协作规则：每日进度同步、接口前置约定、先有后优、风险早报。")

# ============================================================
# 十、风险预案
# ============================================================
add_heading("十、风险预案", 1)
add_table(
    ["风险场景", "应对方案"],
    [
        ["WinCLIP效果不达预期", "替换PatchCore/Anomalib基线，接口保持兼容"],
        ["VLM速度过慢", "切换轻量化模型，或高频场景采用模板化生成"],
        ["RAG匹配不准/引用错误", "优化分块策略，人工预置核心标准条目"],
        ["门控阈值不合理", "中位数+1.5×MAD辅助标定，支持人工微调"],
        ["JSON Schema不服从", "正则提取中间层；成功率低于50%直接降级人工选择"],
        ["真实样本获取困难", "采用公开真实工业数据集+模拟场景干扰做跨域验证"],
        ["工作量爆炸", "对比实验分阶段：中期2组，结题补无门控串行对比"],
    ]
)

# ============================================================
# 十一、参考文献
# ============================================================
add_heading("十一、参考文献", 1)
refs = [
    "[1] Bergmann P, Fauser M, Sattlegger D, et al. MVTec AD-A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection[C]//CVPR. 2019.",
    "[2] Jeong J, Kim S, Kim C, et al. WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation[C]//CVPR. 2023.",
    "[3] Radford A, Kim J W, Hallacy C, et al. Learning Transferable Visual Models From Natural Language Supervision[C]//ICML. 2021.",
    "[4] Lewis P, Perez E, Piktus A, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks[C]//NeurIPS. 2020.",
    "[5] Bai J, Bai S, Yang S, et al. Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities[J]. arXiv preprint, 2023.",
    "[6] GB/T 34528-2017, 金属材料 表面缺陷目视检测导则[S]. 2017.",
    "[7] Wang G, Han G, Li S, et al. A Review of Visual Defect Detection Methods for Industrial Products[J]. Journal of Manufacturing Systems, 2023.",
    "[8] Zhang H, Li J, Wang Y, et al. Semi-supervised Industrial Anomaly Detection: A Survey[J]. IEEE Transactions on Industrial Informatics, 2024.",
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    run = p.add_run(ref)
    run.font.size = Pt(10.5)
    run.font.name = '宋体'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

# 保存
doc.save(OUT_PATH)
print(f"开题报告已生成: {OUT_PATH}")
