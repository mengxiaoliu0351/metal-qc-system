# -*- coding: utf-8 -*-
"""
实验2升级版：纯VLM在30张抽样上的准确率（避免10张样本量不足）
从MVTec metal_nut test集随机抽15正常+15异常
"""
import os, sys, json, random, time
from PIL import Image
ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT,"src"))
from cognition import vlm_api

random.seed(42)
TEST = os.path.join(ROOT,"data","mvtec","metal_nut","test")
OUT = os.path.join(ROOT,"results","experiments")

# 抽样
good = sorted([f for f in os.listdir(os.path.join(TEST,"good")) if f.endswith(".png")])
bad_cats = [c for c in os.listdir(TEST) if c!="good" and os.path.isdir(os.path.join(TEST,c))]
bad = []
for c in bad_cats:
    for f in sorted(os.listdir(os.path.join(TEST,c))):
        if f.endswith(".png"):
            bad.append((c,f))
sample_good = random.sample(good, 15)
sample_bad = random.sample(bad, 15)
print(f"抽样: 15正常 + 15异常 = 30张")

PROMPT = """这是一张金属零件的照片。请判断它是否有表面缺陷。
只输出JSON：{"conclusion":"合格/不合格", "reason":"不超过20字"}
如果看不出明显缺陷，输出"合格"。"""

records = []
t0_all = time.time()
for i, fn in enumerate(sample_good):
    p = os.path.join(TEST,"good",fn)
    out = vlm_api.analyze(p, PROMPT, backend="api")
    pred = "合格" if out.get("conclusion","").startswith("合") else "不合格"
    records.append({"file":fn,"label":"合格","pred":pred,"ok":out.get("ok"),"raw":out.get("raw","")[:200]})
    print(f"  {i+1}/30 good: pred={pred}")

for i,(cat,fn) in enumerate(sample_bad):
    p = os.path.join(TEST,cat,fn)
    out = vlm_api.analyze(p, PROMPT, backend="api")
    pred = "合格" if out.get("conclusion","").startswith("合") else "不合格"
    records.append({"file":f"{cat}/{fn}","label":"不合格","pred":pred,"ok":out.get("ok"),"raw":out.get("raw","")[:200]})
    print(f"  {16+i}/30 bad({cat}): pred={pred}")

dt = time.time()-t0_all
correct = sum(1 for r in records if r["label"]==r["pred"])
acc = correct/len(records)
# 混淆矩阵
TP = sum(1 for r in records if r["label"]=="不合格" and r["pred"]=="不合格")
TN = sum(1 for r in records if r["label"]=="合格" and r["pred"]=="合格")
FP = sum(1 for r in records if r["label"]=="合格" and r["pred"]=="不合格")
FN = sum(1 for r in records if r["label"]=="不合格" and r["pred"]=="合格")
print(f"\n=== 结果 ===")
print(f"30张抽样准确率: {acc:.4f} ({correct}/30)")
print(f"混淆矩阵: TN={TN} FP={FP} FN={FN} TP={TP}")
print(f"耗时: {dt:.1f}s ({dt/30:.2f}s/张)")

out = {"n":30,"accuracy":round(acc,4),"TN":TN,"FP":FP,"FN":FN,"TP":TP,
       "avg_time_s":round(dt/30,2),"records":records}
with open(os.path.join(OUT,"exp2_vlm_30.json"),"w",encoding="utf-8") as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
