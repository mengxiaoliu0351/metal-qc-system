# -*- coding: utf-8 -*-
"""
RAG 标准检索 + 引用ID校验
=========================
复用先导实验的标准片段库（GB/T 34528-2017）与 TF-IDF 检索逻辑，
封装为 KnowledgeBase 类，供 Streamlit 后端调用。
"""
import csv
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# GB/T 34528-2017《金属材料 表面缺陷目视检测导则》标准片段库
STANDARD_FRAGMENTS = [
    {"id": "GB34528-P03-S1", "standard": "GB/T 34528-2017", "page": 3,
     "section": "范围",
     "text": "本标准规定了金属材料表面缺陷目视检测的一般要求、检测方法、缺陷分类及评定规则。本标准适用于黑色金属和有色金属材料及制品的表面缺陷目视检测。"},
    {"id": "GB34528-P05-S2", "standard": "GB/T 34528-2017", "page": 5,
     "section": "缺陷分类-裂纹",
     "text": "裂纹是金属表面最危险的缺陷之一，呈直线或曲线状的连续开裂，深度通常大于宽度。裂纹可分为发纹、纵裂、横裂、龟裂等。裂纹缺陷通常判定为不合格，需进一步无损检测确认深度。"},
    {"id": "GB34528-P06-S3", "standard": "GB/T 34528-2017", "page": 6,
     "section": "缺陷分类-夹杂",
     "text": "夹杂是指金属表面存在的非金属夹杂物暴露，呈点状、条状或不规则形状，颜色与基体不同。夹杂根据大小和数量评定，单个夹杂直径大于1mm或密集分布时判定为不合格。"},
    {"id": "GB34528-P07-S4", "standard": "GB/T 34528-2017", "page": 7,
     "section": "缺陷分类-划痕",
     "text": "划痕是金属表面因机械摩擦产生的线状沟槽，深度一般较浅。划痕长度大于5mm或深度超过0.05mm时影响表面质量，需根据使用场合评定。装饰面划痕通常从严判定。"},
    {"id": "GB34528-P08-S5", "standard": "GB/T 34528-2017", "page": 8,
     "section": "缺陷分类-锈蚀",
     "text": "锈蚀是金属表面与环境介质发生化学或电化学作用产生的腐蚀现象，呈红褐色、暗褐色或黑色斑点/片状。初期锈蚀（浮锈）可清除后使用；深度锈蚀导致截面减薄时判定为不合格。"},
    {"id": "GB34528-P09-S6", "standard": "GB/T 34528-2017", "page": 9,
     "section": "缺陷分类-变形",
     "text": "变形是金属零件的几何形状偏离设计要求，包括弯曲、扭曲、凹陷、凸起等。变形影响装配和使用性能时判定为不合格。弯曲变形可用直尺或塞尺测量挠度，超过公差范围即不合格。"},
    {"id": "GB34528-P10-S7", "standard": "GB/T 34528-2017", "page": 10,
     "section": "检测条件-光照",
     "text": "目视检测应在充足的自然光或等效人工照明条件下进行，被检表面照度不应低于500lx。检测人员应距被检表面300mm~500mm，视线与表面夹角不小于30度。必要时使用放大镜辅助观察。"},
    {"id": "GB34528-P12-S8", "standard": "GB/T 34528-2017", "page": 12,
     "section": "评定规则-一般原则",
     "text": "缺陷评定应结合缺陷类型、尺寸、数量、位置及使用场合综合判定。承载面、密封面、装饰面从严判定；非加工面、非关键面可从宽。所有缺陷均应记录位置、类型和尺寸，出具检测报告。"},
    {"id": "GB34528-P13-S9", "standard": "GB/T 34528-2017", "page": 13,
     "section": "检测报告",
     "text": "检测报告应包括以下内容：检测对象名称及编号、检测依据标准、检测环境条件、缺陷位置示意图、缺陷类型及尺寸、评定结论、检测人员及日期。报告应可追溯，保存期不少于产品使用寿命。"},
    {"id": "GB34528-P15-S10", "standard": "GB/T 34528-2017", "page": 15,
     "section": "缺陷处置建议",
     "text": "不合格缺陷的处置方式包括：返修（打磨、补焊、矫形）、降级使用、报废。返修后应重新检测确认。裂纹缺陷一般不允许补焊返修，需整体评估；夹杂和划痕可打磨清除后复测深度。"},
]


class KnowledgeBase:
    """标准知识库：建库 -> 检索 -> 引用校验"""

    def __init__(self, fragments=None, csv_path=None):
        self.fragments = fragments or STANDARD_FRAGMENTS
        self.texts = [f["text"] for f in self.fragments]
        self.ids = [f["id"] for f in self.fragments]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        self.csv_path = csv_path
        if csv_path:
            self._save_csv()

    def _save_csv(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "standard", "page", "section", "text"])
            writer.writeheader()
            writer.writerows(self.fragments)

    def retrieve(self, query, top_k=3):
        """按缺陷描述检索最相关的标准片段，返回带分数与页码的列表"""
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            frag = self.fragments[idx]
            results.append({
                "id": frag["id"],
                "score": float(sims[idx]),
                "standard": frag["standard"],
                "page": frag["page"],
                "section": frag["section"],
                "text": frag["text"],
            })
        return results

    def validate_ref_ids(self, output_ids, retrieved_ids):
        """后处理一致性校验：输出引用ID是否都在本次检索结果中"""
        valid_ids = [rid for rid in output_ids if rid in retrieved_ids]
        invalid_ids = [rid for rid in output_ids if rid not in retrieved_ids]
        return {
            "all_valid": len(invalid_ids) == 0,
            "valid_ids": valid_ids,
            "invalid_ids": invalid_ids,
        }


if __name__ == "__main__":
    kb = KnowledgeBase(csv_path=r"D:\programme file\ResearchArticle\results\cognition\standard_library.csv")
    print(f"知识库初始化完成：{len(kb.fragments)} 条标准片段")
    r = kb.retrieve("表面有一条线状划痕，长度约3-5mm，深度较浅", top_k=3)
    for item in r:
        print(f"  [{item['id']}] P{item['page']} {item['section']} score={item['score']:.4f}")
    v = kb.validate_ref_ids(["GB34528-P07-S4", "GB34528-P99-S99"],
                            [x["id"] for x in r])
    print(f"校验: all_valid={v['all_valid']}, 无效ID={v['invalid_ids']}")
