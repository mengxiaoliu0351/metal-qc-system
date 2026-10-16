# -*- coding: utf-8 -*-
"""生成中期报告 Word"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import os

OUT = r"D:\programme file\ResearchArticle\中期报告_双链路智能质检辅助系统.docx"
EXP = r"D:\programme file\ResearchArticle\results\experiments"

doc = Document()

# 全局中文字体
style = doc.styles["Normal"]
style.font.name = "宋体"
style.font.size = Pt(11)
style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

def set_cn(run, font="宋体", size=11, bold=False, color=None):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    if color:
        run.font.color.rgb = RGBColor(*color)

def h1(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_cn(r, "黑体", 15, True, (0x1f,0x3a,0x5f))
    p.space_after = Pt(6)

def h2(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_cn(r, "黑体", 12.5, True, (0x2c,0x5a,0x8f))

def body(text):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(22)
    p.paragraph_format.line_spacing = 1.4
    r = p.add_run(text)
    set_cn(r, "宋体", 11)

def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    set_cn(r, "宋体", 11)

# ===== 封面标题 =====
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("面向金属表面缺陷的双链路智能质检辅助系统")
set_cn(r, "黑体", 20, True, (0x1f,0x3a,0x5f))
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("——《人工智能应用实践》中期报告")
set_cn(r, "楷体", 13)
info = doc.add_paragraph()
info.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = info.add_run("山东大学 人工智能学院")
set_cn(r, "宋体", 11)
doc.add_paragraph()

# ===== 一、问题发现 =====
h1("一、问题发现与研究动机")
body("工业金属零件在量产中普遍存在划痕、弯曲、色差、翻面等表面缺陷。传统人工目检存在疲劳漏检、一致性差、成本高的问题；而纯通用视觉大模型（VLM）虽能给出自然语言描述，却存在推理慢（约0.9秒/张）、API成本高、对几何类异常（如零件翻面）不敏感等缺陷。")
body("本项目针对上述痛点，设计并实现一套“无监督异常检出 + 有监督类型归因 + VLM语义推理 + 国标RAG检索 + 人在回路持续学习”的双链路智能质检辅助系统，目标是在保证检出率的同时大幅降低大模型调用成本，并具备可解释、可追溯、可进化的能力。")

# ===== 二、现状调研 =====
h1("二、现状调研")
bullet("无监督异常检测：以PaDiM、PatchCore、WinCLIP为代表，仅需正常样本即可建模，无需缺陷标注，适合缺陷样本稀缺的产线。")
bullet("零样本/小样本分类：CLIP类多模态模型可通过文本提示实现缺陷类型分类，但对纹理细粒度差异不敏感。")
bullet("大模型视觉推理（VLM）：Qwen-VL、GPT-4V等可直接输出自然语言判断与建议，但推理慢、成本高、存在黑盒问题。")
bullet("人在回路持续学习：从人工反馈中迭代模型（隐式反馈学习），是提升系统在真实产线适应性的关键路径。")

# ===== 三、系统方案 =====
h1("三、系统总体方案")
h2("3.1 双链路门控架构")
body("系统采用“快链路—慢链路”门控架构：对待检图像先由本地WinCLIP模型提取patch级特征，与该产品正常样本记忆库比对，计算异常分数。若分数低于阈值，直接判定合格并秒级返回（快链路）；若分数高于阈值，则触发慢链路，调用VLM进行视觉推理、结合GB/T 34528国标条文的语义检索，输出缺陷描述与处置建议。")
h2("3.2 多产品独立记忆库")
body("针对不同金属零件分别建立正常样本记忆库并独立标定阈值，避免跨产品分数漂移导致的误报。目前已建成三个产品库：")

# 表：三产品
tbl = doc.add_table(rows=1, cols=4)
tbl.style = "Light Grid Accent 1"
hdr = tbl.rows[0].cells
for i, t in enumerate(["产品", "正常样本数", "标定阈值", "阈值方法"]):
    hdr[i].text = ""
    rr = hdr[i].paragraphs[0].add_run(t); set_cn(rr, "黑体", 10.5, True)
rows = [
    ("金属螺母 metal_nut", "80", "0.2345", "正常样本中位数+1.5×MAD"),
    ("金属螺丝 screw", "240", "0.1788", "独立留出标定"),
    ("金属网格 grid", "184", "0.2311", "独立留出标定"),
]
for row in rows:
    cells = tbl.add_row().cells
    for i, v in enumerate(row):
        cells[i].text = ""
        rr = cells[i].paragraphs[0].add_run(v); set_cn(rr, "宋体", 10.5)
doc.add_paragraph()

# ===== 四、创新点 =====
h1("四、核心创新点")
bullet("双链路门控：以本地无监督检出为主、VLM为辅，实测将VLM调用量降低约81%，同时保持高检出率。")
bullet("有监督缺陷类型分类头：在CLIP整图特征上训练标准化逻辑回归分类器，对金属螺母5类缺陷（正常/弯曲/色差/翻面/划痕）测试准确率达89.66%，纠正了零样本分类对弯曲件误判为“正常”的问题。")
bullet("人在回路在线回灌（持续学习）：人工反馈“判错”后，系统自动将该困难样本的patch特征并入对应产品记忆库并落盘，实现模型随使用持续进化。")
bullet("国标语义RAG：以CLIP文本编码器将GB/T 34528条文与缺陷描述在同一多模态语义空间匹配，替代原基于关键词分数的低效检索，使慢链路结论可追溯到具体标准条文。")

# ===== 五、实验结果 =====
h1("五、实验与结果")
h2("5.1 主实验检出性能")
body("在MVTec-AD metal_nut测试集（115张：正常22张、异常93张）上，以正常样本标定阈值0.2345：整体准确率96.52%，AUROC达0.9868。混淆矩阵为：正常件中21张判对、1张误报；异常件中90张判对、3张漏检。")

# 混淆矩阵表
tbl2 = doc.add_table(rows=1, cols=3); tbl2.style = "Light Grid Accent 1"
c = tbl2.rows[0].cells
for i,t in enumerate(["", "预测正常", "预测异常"]):
    c[i].text=""; rr=c[i].paragraphs[0].add_run(t); set_cn(rr,"黑体",10.5,True)
for row in [("实际正常","21","1（误报）"),("实际异常","3（漏检）","90")]:
    cc = tbl2.add_row().cells
    for i,v in enumerate(row):
        cc[i].text=""; rr=cc[i].paragraphs[0].add_run(v); set_cn(rr,"宋体",10.5)
doc.add_paragraph()

h2("5.2 方法选型对比（自研实验）")
body("为验证异常度量的选型合理性，本项目在同一批CLIP patch特征上对比了三种经典距离度量：最近邻余弦、kNN-5余弦、PaDiM马氏距离（PCA降维+LedoitWolf收缩）。结果如下表及图所示，最近邻余弦在该特征空间表现最优（98.68%），马氏距离因高维协方差估计退化而显著下降，证明本系统选用最近邻余弦是经过实验对比的合理决策，而非盲目调用。")

# 图1
if os.path.exists(EXP + r"\fig_metric_comparison.png"):
    doc.add_picture(EXP + r"\fig_metric_comparison.png", width=Inches(5.6))
    cap = doc.paragraphs[-1]; cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

h2("5.3 三方案对比")
body("纯WinCLIP、纯VLM、双链路门控三方案对比表明：双链路方案在准确率（96.52%）、推理速度（约0.23秒/张快链路）、成本（仅19%图像调用VLM）三方面取得平衡。VLM抽样10张准确率90%，其短板在于对“翻面”这类几何异常漏判，而WinCLIP能稳定检出。")
if os.path.exists(EXP + r"\fig_three_way.png"):
    doc.add_picture(EXP + r"\fig_three_way.png", width=Inches(5.6))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

h2("5.4 域偏移测试")
body("为验证“独立建库+独立阈值”的必要性，将metal_nut记忆库直接用于NEU-DET钢带缺陷图（120张）：分数从metal_nut正常件的均值0.20整体抬高到0.48，超过原阈值比例达100%，即跨域会全部误报。这量化证明了不能跨产品复用记忆库，必须按产线产品独立建模。")

h2("5.5 持续学习验证")
body("触发一次人工“判错”反馈后，对应产品记忆库特征数由15680增长至15876（单张图196个patch特征并入并落盘），验证了人在回路在线回灌机制真实生效。")

# ===== 六、中期进展 =====
h1("六、中期进展与可演示能力")
bullet("已完成：FastAPI后端 + Web前端（检测/批量/历史详情/PDF报告/导出Excel）、三产品记忆库、双链路门控、监督分类头、VLM慢链路、国标语义RAG、AI对话助手、人在回路在线回灌。")
bullet("可现场演示：上传零件图即时检测、热力图可视化、缺陷类型与置信度、慢链路VLM分析与标准引用、反馈后系统自动学习。")

# ===== 七、不足与下一步 =====
h1("七、不足与下一步计划")
bullet("泛化性：当前基于公开数据集验证，真实产线相机/光源下仍需按产品独立建库校准。")
bullet("分类头覆盖：目前监督分类器仅针对metal_nut训练，下一步将为screw、grid补训分类头。")
bullet("对比基线：后续将补充PatchCore/PaDiM在同数据集上的系统对比，进一步量化本方法相对经典无监督方法的优势。")
bullet("结题目标：完善真实产线迁移论证、撰写结题报告并优化答辩展示。")

doc.save(OUT)
print("已生成:", OUT)
