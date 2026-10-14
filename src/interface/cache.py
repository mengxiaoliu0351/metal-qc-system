# -*- coding: utf-8 -*-
"""
缓存优先逻辑：样例图结果预缓存，现场0秒出结果
=============================================
缓存目录：results/demo_cache/
每个样例一个子目录 <image_stem>/，内含：
    input.png     原图
    heatmap.png   热力图
    result.json   完整结果（分数/门控/VLM/RAG/校验）
"""
import hashlib
import json
import os

CACHE_DIR = r"D:\programme file\ResearchArticle\results\demo_cache"


def cache_key(image_path: str) -> str:
    """缓存目录名：样例图用文件名，其他用内容哈希"""
    # 样例目录下的文件按名缓存（预生成），其余按内容哈希
    sample_dir = r"D:\programme file\ResearchArticle\data\demo_samples"
    if os.path.dirname(os.path.abspath(image_path)) == os.path.abspath(sample_dir):
        return os.path.splitext(os.path.basename(image_path))[0]
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def cache_path(image_path: str) -> str:
    return os.path.join(CACHE_DIR, cache_key(image_path))


def load_cached(image_path: str):
    """读取缓存，返回 dict 或 None"""
    d = cache_path(image_path)
    result_file = os.path.join(d, "result.json")
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(image_path: str, result: dict):
    """写入缓存（result 需含 input/heatmap 路径）"""
    d = cache_path(image_path)
    os.makedirs(d, exist_ok=True)
    # 复制原图和热力图到缓存目录（若 result 中带路径）
    for key in ("input_png", "heatmap_png"):
        src = result.get(key)
        if src and os.path.exists(src):
            dst = os.path.join(d, os.path.basename(src))
            if not os.path.exists(dst):
                from shutil import copyfile
                copyfile(src, dst)
            result[key] = dst
    with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
