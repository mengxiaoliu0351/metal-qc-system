# -*- coding: utf-8 -*-
"""
认知层先导实验：RAG 标准检索 + 引用 ID 可溯源校验
====================================================
核心思路：
1. 构建金属表面缺陷质检标准片段库（基于 GB/T 34528 等标准内容），每个片段分配唯一引用 ID
2. 用 TF-IDF + 余弦相似度做 RAG 检索（轻量、零额外依赖，开题验证机制用）
3. 对缺陷图生成自然语言描述（基于缺陷类型+热力图位置的规则生成，模拟 VLM 输出）
4. 模拟 VLM 结构化输出：结论 + 缺陷描述 + 引用 ID 列表
5. 后处理一致性校验：检查引用 ID 是否在本次检索结果中
6. 展示校验通过 vs 校验失败（编造引用）的对比

输出：
- results/cognition/standard_library.csv  标准片段库
- results/cognition/retrieval_result.png   RAG 检索结果展示
- results/cognition/ref_validation_pass.png  引用校验通过样例
- results/cognition/ref_validation_fail.png  引用校验失败（编造）样例
- results/cognition/cognition_demo.png      认知层完整流程演示
"""
import os
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

ROOT = r"D:\programme file\ResearchArticle"
OUT_DIR = os.path.join(ROOT, "results", "cognition")
os.makedirs(OUT_DIR, exist_ok=True)

# ============ 1. 构建标准片段库 ============
# 基于 GB/T 34528-2017《金属材料 表面缺陷目视检测导则》及金属表面缺陷质检常识
STANDARD_FRAGMENTS = [
    {
        "id": "GB34528-P03-S1",
        "standard": "GB/T 34528-2017",
        "page": 3,
        "section": "范围",
        "text": "本标准规定了金属材料表面缺陷目视检测的一般要求、检测方法、缺陷分类及评定规则。本标准适用于黑色金属和有色金属材料及制品的表面缺陷目视检测。",
    },
    {
        "id": "GB34528-P05-S2",
        "standard": "GB/T 34528-2017",
        "page": 5,
        "section": "缺陷分类-裂纹",
        "text": "裂纹是金属表面最危险的缺陷之一，呈直线或曲线状的连续开裂，深度通常大于宽度。裂纹可分为发纹、纵裂、横裂、龟裂等。裂纹缺陷通常判定为不合格，需进一步无损检测确认深度。",
    },
    {
        "id": "GB34528-P06-S3",
        "standard": "GB/T 34528-2017",
        "page": 6,
        "section": "缺陷分类-夹杂",
        "text": "夹杂是指金属表面存在的非金属夹杂物暴露，呈点状、条状或不规则形状，颜色与基体不同。夹杂根据大小和数量评定，单个夹杂直径大于1mm或密集分布时判定为不合格。",
    },
    {
        "id": "GB34528-P07-S4",
        "standard": "GB/T 34528-2017",
        "page": 7,
        "section": "缺陷分类-划痕",
        "text": "划痕是金属表面因机械摩擦产生的线状沟槽，深度一般较浅。划痕长度大于5mm或深度超过0.05mm时影响表面质量，需根据使用场合评定。装饰面划痕通常从严判定。",
    },
    {
        "id": "GB34528-P08-S5",
        "standard": "GB/T 34528-2017",
        "page": 8,
        "section": "缺陷分类-锈蚀",
        "text": "锈蚀是金属表面与环境介质发生化学或电化学作用产生的腐蚀现象，呈红褐色、暗褐色或黑色斑点/片状。初期锈蚀（浮锈）可清除后使用；深度锈蚀导致截面减薄时判定为不合格。",
    },
    {
        "id": "GB34528-P09-S6",
        "standard": "GB/T 34528-2017",
        "page": 9,
        "section": "缺陷分类-变形",
        "text": "变形是金属零件的几何形状偏离设计要求，包括弯曲、扭曲、凹陷、凸起等。变形影响装配和使用性能时判定为不合格。弯曲变形可用直尺或塞尺测量挠度，超过公差范围即不合格。",
    },
    {
        "id": "GB34528-P10-S7",
        "standard": "GB/T 34528-2017",
        "page": 10,
        "section": "检测条件-光照",
        "text": "目视检测应在充足的自然光或等效人工照明条件下进行，被检表面照度不应低于500lx。检测人员应距被检表面300mm~500mm，视线与表面夹角不小于30度。必要时使用放大镜辅助观察。",
    },
    {
        "id": "GB34528-P12-S8",
        "standard": "GB/T 34528-2017",
        "page": 12,
        "section": "评定规则-一般原则",
        "text": "缺陷评定应结合缺陷类型、尺寸、数量、位置及使用场合综合判定。承载面、密封面、装饰面从严判定；非加工面、非关键面可从宽。所有缺陷均应记录位置、类型和尺寸，出具检测报告。",
    },
    {
        "id": "GB34528-P13-S9",
        "standard": "GB/T 34528-2017",
        "page": 13,
        "section": "检测报告",
        "text": "检测报告应包括以下内容：检测对象名称及编号、检测依据标准、检测环境条件、缺陷位置示意图、缺陷类型及尺寸、评定结论、检测人员及日期。报告应可追溯，保存期不少于产品使用寿命。",
    },
    {
        "id": "GB34528-P15-S10",
        "standard": "GB/T 34528-2017",
        "page": 15,
        "section": "缺陷处置建议",
        "text": "不合格缺陷的处置方式包括：返修（打磨、补焊、矫形）、降级使用、报废。返修后应重新检测确认。裂纹缺陷一般不允许补焊返修，需整体评估；夹杂和划痕可打磨清除后复测深度。",
    },
]

# 保存标准库
csv_path = os.path.join(OUT_DIR, "standard_library.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "standard", "page", "section", "text"])
    writer.writeheader()
    writer.writerows(STANDARD_FRAGMENTS)
print(f"标准片段库已保存: {csv_path} ({len(STANDARD_FRAGMENTS)} 条)")

# ============ 2. TF-IDF RAG 检索 ============
class SimpleRAG:
    def __init__(self, fragments):
        self.fragments = fragments
        self.texts = [f["text"] for f in fragments]
        self.ids = [f["id"] for f in fragments]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=5000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "id": self.ids[idx],
                "score": float(sims[idx]),
                "fragment": self.fragments[idx],
            })
        return results

rag = SimpleRAG(STANDARD_FRAGMENTS)
print("RAG 检索引擎初始化完成（TF-IDF + 余弦相似度）")

# ============ 3. 模拟缺陷描述生成（模拟 VLM 输出） ============
def generate_defect_description(defect_type, location_hint="中心区域"):
    """基于缺陷类型生成自然语言描述，模拟 VLM 对裁剪缺陷图的描述输出"""
    templates = {
        "scratch": f"在零件{location_hint}可见一条明显的线状划痕，长度约3-5mm，深度较浅，沿轴向延伸。划痕周围基体完整，无其他伴随缺陷。",
        "bent": f"零件{location_hint}存在可见的弯曲变形，整体轴线偏离直线，挠度较大。变形区域表面无裂纹，但几何形状明显偏离设计要求。",
        "color": f"零件{location_hint}表面存在颜色异常区域，呈局部色差/氧化变色，与周围基体颜色明显不同。变色区域边界较清晰，表面粗糙度无明显变化。",
        "flip": f"零件{location_hint}存在翻转/错位变形，特征位置与标准方位不符。整体形状异常，装配定位面偏移。",
        "crazing": f"零件{location_hint}表面可见网状裂纹（龟裂），呈细密的网状分布，裂纹深度较浅但分布密集。",
        "inclusion": f"零件{location_hint}表面可见点状非金属夹杂，颜色与基体不同，单个夹杂直径约0.5-1mm，分布较分散。",
    }
    return templates.get(defect_type, f"零件{location_hint}表面存在异常区域，需进一步检测确认缺陷类型。")

# ============ 4. 引用 ID 校验机制 ============
def validate_ref_ids(output_ids, retrieved_ids):
    """后处理一致性校验：检查输出的引用 ID 是否都在检索结果中"""
    valid_ids = []
    invalid_ids = []
    for rid in output_ids:
        if rid in retrieved_ids:
            valid_ids.append(rid)
        else:
            invalid_ids.append(rid)
    return {
        "all_valid": len(invalid_ids) == 0,
        "valid_ids": valid_ids,
        "invalid_ids": invalid_ids,
    }

def format_vlm_output(conclusion, defect_desc, ref_ids, suggestion):
    """模拟 VLM JSON 结构化输出"""
    return {
        "conclusion": conclusion,
        "defect_desc": defect_desc,
        "ref_ids": ref_ids,
        "suggestion": suggestion,
    }

# ============ 5. 运行完整认知流程 ============
print("\n=== 认知层实验开始 ===")

# 测试案例1：划痕缺陷（校验通过）
print("\n--- 案例1：划痕缺陷（引用校验通过） ---")
desc1 = generate_defect_description("scratch", "右上区域")
print(f"缺陷描述: {desc1[:50]}...")
retrieved1 = rag.retrieve(desc1, top_k=3)
print("RAG 检索 Top-3:")
for r in retrieved1:
    print(f"  [{r['id']}] 相似度={r['score']:.4f} - {r['fragment']['section']}")

# VLM 输出正确的引用 ID（从检索结果中选）
vlm_output1 = format_vlm_output(
    conclusion="不合格",
    defect_desc=desc1,
    ref_ids=[retrieved1[0]["id"], retrieved1[3-1]["id"] if len(retrieved1)>=3 else retrieved1[0]["id"]],
    suggestion="建议打磨清除划痕后复测深度，若深度超过0.05mm则返修。",
)
validation1 = validate_ref_ids(vlm_output1["ref_ids"], [r["id"] for r in retrieved1])
print(f"VLM 输出引用 ID: {vlm_output1['ref_ids']}")
print(f"校验结果: {'全部通过 ✓' if validation1['all_valid'] else '存在无效 ID ✗'}")

# 测试案例2：编造引用（校验失败）
print("\n--- 案例2：编造引用（校验失败，触发降级） ---")
desc2 = generate_defect_description("bent", "中心区域")
retrieved2 = rag.retrieve(desc2, top_k=3)
print("RAG 检索 Top-3:")
for r in retrieved2:
    print(f"  [{r['id']}] 相似度={r['score']:.4f} - {r['fragment']['section']}")

# VLM 编造了一个不存在的引用 ID
vlm_output2 = format_vlm_output(
    conclusion="不合格",
    defect_desc=desc2,
    ref_ids=[retrieved2[0]["id"], "GB34528-P99-S99"],  # 第二个是编造的
    suggestion="建议矫形返修，返修后重新检测几何公差。",
)
validation2 = validate_ref_ids(vlm_output2["ref_ids"], [r["id"] for r in retrieved2])
print(f"VLM 输出引用 ID: {vlm_output2['ref_ids']}")
print(f"校验结果: {'全部通过 ✓' if validation2['all_valid'] else '存在无效 ID ✗'}")
print(f"无效 ID: {validation2['invalid_ids']}")
print("→ 自动标红提示「引用待核验」，降级为展示原始标准片段供人工选择")

# ============ 6. 生成可视化结果 ============

# 6.1 RAG 检索结果展示
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")
ax.set_title("RAG 标准检索演示（划痕缺陷描述 → Top-3 标准片段）", fontsize=14, fontweight="bold", pad=20)

# 查询文本
query_box = FancyBboxPatch((0.02, 0.78), 0.96, 0.16, boxstyle="round,pad=0.02",
                             facecolor="#e8f4fd", edgecolor="#2196F3", linewidth=2)
ax.add_patch(query_box)
ax.text(0.04, 0.88, "【缺陷描述（VLM 生成）】", fontsize=11, fontweight="bold", color="#1565C0")
ax.text(0.04, 0.82, desc1[:80] + "...", fontsize=10, color="#333")

# 检索结果
colors = ["#fff3e0", "#fff8e1", "#fffde7"]
for i, r in enumerate(retrieved1):
    y_pos = 0.62 - i * 0.22
    box = FancyBboxPatch((0.02, y_pos), 0.96, 0.18, boxstyle="round,pad=0.02",
                          facecolor=colors[i], edgecolor="#FF9800", linewidth=1.5)
    ax.add_patch(box)
    ax.text(0.04, y_pos + 0.13, f"Top-{i+1}  引用ID: {r['id']}  |  标准: {r['fragment']['standard']}  |  页码: 第{r['fragment']['page']}页  |  相似度: {r['score']:.4f}",
            fontsize=10, fontweight="bold", color="#E65100")
    ax.text(0.04, y_pos + 0.05, r["fragment"]["text"][:75] + "...", fontsize=9, color="#555")

plt.tight_layout()
retrieval_path = os.path.join(OUT_DIR, "retrieval_result.png")
plt.savefig(retrieval_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nRAG 检索结果图已保存: {retrieval_path}")

# 6.2 引用校验通过样例
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis("off")
ax.set_title("引用 ID 校验 —— 通过样例", fontsize=14, fontweight="bold", color="#2E7D32", pad=15)

# VLM 输出
box1 = FancyBboxPatch((0.02, 0.55), 0.96, 0.35, boxstyle="round,pad=0.02",
                       facecolor="#f1f8e9", edgecolor="#4CAF50", linewidth=2)
ax.add_patch(box1)
ax.text(0.04, 0.82, "【VLM 结构化输出（JSON）】", fontsize=11, fontweight="bold", color="#33691E")
ax.text(0.04, 0.74, f'conclusion: "{vlm_output1["conclusion"]}"', fontsize=10)
ax.text(0.04, 0.67, f'ref_ids: {vlm_output1["ref_ids"]}', fontsize=10)
ax.text(0.04, 0.60, f'suggestion: "{vlm_output1["suggestion"][:30]}..."', fontsize=10)

# 校验结果
box2 = FancyBboxPatch((0.02, 0.08), 0.96, 0.38, boxstyle="round,pad=0.02",
                       facecolor="#e8f5e9", edgecolor="#66BB6A", linewidth=2)
ax.add_patch(box2)
ax.text(0.04, 0.38, "【后处理一致性校验】", fontsize=11, fontweight="bold", color="#1B5E20")
ax.text(0.04, 0.30, f"检索返回 ID: {[r['id'] for r in retrieved1]}", fontsize=10)
ax.text(0.04, 0.22, f"VLM 输出 ID: {vlm_output1['ref_ids']}", fontsize=10)
ax.text(0.04, 0.13, "[校验通过] 所有引用 ID 均在检索结果中，标注标准号与页码后正常展示",
        fontsize=11, fontweight="bold", color="#2E7D32")

plt.tight_layout()
pass_path = os.path.join(OUT_DIR, "ref_validation_pass.png")
plt.savefig(pass_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"引用校验通过样例已保存: {pass_path}")

# 6.3 引用校验失败样例
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis("off")
ax.set_title("引用 ID 校验 —— 失败（编造引用）触发降级", fontsize=14, fontweight="bold", color="#C62828", pad=15)

box1 = FancyBboxPatch((0.02, 0.55), 0.96, 0.35, boxstyle="round,pad=0.02",
                       facecolor="#ffebee", edgecolor="#EF5350", linewidth=2)
ax.add_patch(box1)
ax.text(0.04, 0.82, "【VLM 结构化输出（JSON）】", fontsize=11, fontweight="bold", color="#B71C1C")
ax.text(0.04, 0.74, f'conclusion: "{vlm_output2["conclusion"]}"', fontsize=10)
ax.text(0.04, 0.67, f'ref_ids: {vlm_output2["ref_ids"]}', fontsize=10)
ax.text(0.04, 0.60, f'  → GB34528-P99-S99 是 VLM 编造的不存在的 ID', fontsize=9, color="#D32F2F", style="italic")

box2 = FancyBboxPatch((0.02, 0.08), 0.96, 0.38, boxstyle="round,pad=0.02",
                       facecolor="#fff3e0", edgecolor="#FF9800", linewidth=2)
ax.add_patch(box2)
ax.text(0.04, 0.38, "【后处理一致性校验 → 降级模式】", fontsize=11, fontweight="bold", color="#E65100")
ax.text(0.04, 0.30, f"检索返回 ID: {[r['id'] for r in retrieved2]}", fontsize=10)
ax.text(0.04, 0.22, f"无效 ID: {validation2['invalid_ids']}", fontsize=10, color="#D32F2F", fontweight="bold")
ax.text(0.04, 0.13, "[校验失败] 自动标红「引用待核验」，降级为展示 RAG 原始标准片段，由人工选择适用条款",
        fontsize=11, fontweight="bold", color="#E65100")

plt.tight_layout()
fail_path = os.path.join(OUT_DIR, "ref_validation_fail.png")
plt.savefig(fail_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"引用校验失败样例已保存: {fail_path}")

# 6.4 认知层完整流程演示
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("认知层完整流程：缺陷描述 → RAG 检索 → 引用校验 → 结构化输出", fontsize=14, fontweight="bold")

steps = [
    ("① VLM 缺陷描述", desc1[:60] + "...", "#e3f2fd", "#1565C0"),
    ("② RAG 标准检索", f"Top-1: {retrieved1[0]['id']}\n相似度: {retrieved1[0]['score']:.4f}\n{retrieved1[0]['fragment']['section']}", "#fff3e0", "#E65100"),
    ("③ 引用 ID 校验", f"输出 ID: {vlm_output1['ref_ids']}\n校验: [全部通过]\n所有 ID 均在检索结果中", "#e8f5e9", "#2E7D32"),
    ("④ 结构化输出", f"结论: {vlm_output1['conclusion']}\n引用: {vlm_output1['ref_ids']}\n建议: {vlm_output1['suggestion'][:25]}...", "#f3e5f5", "#6A1B9A"),
]

for ax, (title, content, bg, color) in zip(axes, steps):
    ax.axis("off")
    box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8, boxstyle="round,pad=0.03",
                         facecolor=bg, edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(0.5, 0.78, title, fontsize=12, fontweight="bold", color=color, ha="center")
    ax.text(0.1, 0.45, content, fontsize=9, color="#333", va="center", wrap=True)

# 箭头
for i in range(3):
    axes[i].annotate("", xy=(1.15, 0.5), xytext=(1.05, 0.5),
                     arrowprops=dict(arrowstyle="->", color="#666", lw=2))

plt.tight_layout()
demo_path = os.path.join(OUT_DIR, "cognition_demo.png")
plt.savefig(demo_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"认知层完整流程演示图已保存: {demo_path}")

print("\n=== 认知层先导实验全部完成 ===")
print(f"输出目录: {OUT_DIR}")
