# -*- coding: utf-8 -*-
"""
RAG 标准检索 + 引用ID校验
=========================
检索项目编写的教学知识卡，保留明确来源类型，
封装为 KnowledgeBase 类，供 Streamlit 后端调用。
"""
import csv
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# No unverified standard body text or page numbers are shipped.
from cognition.knowledge_sources import PROJECT_NOTES
STANDARD_FRAGMENTS = PROJECT_NOTES



class KnowledgeBase:
    """标准知识库：建库 -> 检索 -> 引用校验"""

    def __init__(self, fragments=None, csv_path=None):
        self.fragments = fragments or STANDARD_FRAGMENTS
        self.texts = [f.get('section', '') + ' ' + f['text'] for f in self.fragments]
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
            writer.writerows({k: f.get(k) for k in ("id", "standard", "page", "section", "text")} for f in self.fragments)

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
                "text": frag["text"], "source_type": frag.get("source_type"),
                "verified_standard": frag.get("verified_standard", False),
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


class SemanticKnowledgeBase:
    """#3 CLIP 语义检索：条文与缺陷描述在同一多模态空间算余弦相似度。

    encode_fn: callable(list[str]) -> np.ndarray[n, d]，返回已归一化的行向量。
    通常传 open_clip 的文本编码器，这样图像/文本/条文共享同一语义空间。
    """

    def __init__(self, encode_fn, fragments=None):
        self.fragments = fragments or STANDARD_FRAGMENTS
        self.encode_fn = encode_fn
        # 建库时把所有条文编码一次并归一化
        self._matrix = self._norm(encode_fn([f["text"] for f in self.fragments]))

    @staticmethod
    def _norm(mat):
        mat = np.asarray(mat, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        return mat / norms

    def retrieve(self, query, top_k=3):
        q = self._norm(self.encode_fn([query]))[0]
        sims = self._matrix @ q
        top = np.argsort(sims)[::-1][:top_k]
        out = []
        for idx in top:
            f = self.fragments[idx]
            out.append({"id": f["id"], "score": float(sims[idx]),
                        "standard": f["standard"], "page": f["page"],
                        "section": f["section"], "text": f["text"], "source_type": f.get("source_type"),
                        "verified_standard": f.get("verified_standard", False)})
        return out
