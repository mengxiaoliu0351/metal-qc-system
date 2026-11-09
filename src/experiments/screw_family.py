"""Frozen three-arm source/view-family regression and synthetic-light stress test."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from perception.family import FamilyModel, conditional_features, unit
from perception.feature_backends import binary_probability, export_binary, representative_bank
from review.workflow import digest, file_hash, image_hash, read, write, measurement

M2 = "product_a269405b7b9a"
BUILD = "13ff4b2995df4239a4d16d133c48ef6b"
CONDITIONS = {"original": {}, "dim": {"brightness": .65}, "warm": {"rgb": [1.12, 1.0, .82]}}
ARMS = ("separate", "pooled", "structured")


def portable(path):
    return str(path).replace("\\", "/")


def prepare(output):
    groups = [{"key": "mvtec_screw", "name": "MVTec 螺丝视图", "source": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"},
              {"key": "m2_head", "name": "M2 螺钉头部俯视", "source": "https://data.mendeley.com/datasets/2y6mcsyrpy/2"}]
    old = read(ROOT / "results/experiments/perception_upgrade_v1/screw_split.json")
    m2 = read(ROOT / f"results/product_catalog/{M2}/builds/{BUILD}/input.json")
    sources = [dict(bank=old["normal_memory"], train=old["train"],
                    calibration=old["validation"] + old["normal_calibration"], test=old["test"]),
               {role: [s["path"] for s in m2["samples"] if s["role"] == role]
                for role in ("bank", "train", "calibration", "test")}]
    samples, identities = [], {}
    for group, roles in enumerate(sources):
        for role, paths in roles.items():
            for path in paths:
                path = portable(path)
                fingerprint = image_hash(ROOT / path)
                label = 0 if (ROOT / path).parent.name == "good" else 1
                if fingerprint in identities:
                    raise ValueError("Repeated image across frozen partitions: " + path)
                identities[fingerprint] = (role, label)
                samples.append({"path": path, "pixel_hash": fingerprint, "file_sha256": file_hash(ROOT / path),
                                "group": group, "role": role, "label": label})
    protocol = {"version": "screw-family-v1", "groups": groups, "samples": samples,
                "seed": 20261106, "coreset_per_group": 1024, "classifier_C": .1,
                "conditions": CONDITIONS, "training_conditions": ["original"],
                "calibration_objective": "equal-source mean balanced accuracy; ties nearest 0.5; no test tuning",
                "primary": "labelled whole-image classification; identical frozen backbone and training images across arms",
                "memory_secondary": "normal-only patch scores; 99th normal calibration percentile",
                "separate": "known source/view; individual classifier and reference subset",
                "pooled": "one classifier, one threshold, fully mixed normal reference bank",
                "structured": "one conditional classifier, automatic reference routing, grouped views in one bank file",
                "safety": "pilot only; no automatic archival; unknown scope routed to review",
                "acceptance": {"max_original_group_recall_drop": .02, "max_original_macro_accuracy_drop": .02},
                "scope": "two source/view groups, not verified SKU labels; historic retained partitions are regression; lighting is synthetic; physical IDs unavailable",
                "source_artifacts": {"mvtec_split": file_hash(ROOT / "results/experiments/perception_upgrade_v1/screw_split.json"),
                    "m2_input": file_hash(ROOT / f"results/product_catalog/{M2}/builds/{BUILD}/input.json"),
                    "weights": file_hash(ROOT / "weights/CLIP-ViT-B-16-open_clip_model.safetensors")}}
    target = output / "protocol.json"
    if target.exists() and read(target) != protocol: raise ValueError("Frozen protocol changed; choose a new experiment directory")
    write(target, protocol)
    print(json.dumps({"prepared": len(samples), "counts": {str(g): dict(Counter(s["role"] for s in samples if s["group"] == g)) for g in range(2)} }), flush=True)


def transform(image, condition):
    image = image.convert("RGB")
    spec = CONDITIONS[condition]
    if "brightness" in spec: image = ImageEnhance.Brightness(image).enhance(spec["brightness"])
    if "rgb" in spec:
        image = Image.fromarray(np.clip(np.asarray(image, dtype=np.float32) * np.asarray(spec["rgb"]), 0, 255).astype(np.uint8))
    return image


class Features:
    def __init__(self, output, protocol):
        self.output, self.protocol = output, protocol
        if file_hash(ROOT / "weights/CLIP-ViT-B-16-open_clip_model.safetensors") != protocol['source_artifacts']['weights']:
            raise ValueError('Shared backbone changed after protocol freeze')
        self.cache = ROOT / protocol['feature_cache'] if protocol.get('feature_cache') else output / "features"
        if not self.cache.resolve().is_relative_to((ROOT/'results').resolve()):raise ValueError('Feature cache must stay within results')
        self.detector = None
        self.stamp = digest({"weights": protocol["source_artifacts"]["weights"], "conditions": CONDITIONS,
                             "extractor": file_hash(ROOT / "src/perception/winclip_patch.py")})

    def path(self, sample, condition):
        return self.cache / (digest([self.stamp, sample["pixel_hash"], condition]) + ".npz")

    def ensure(self, samples, condition):
        missing = [s for s in samples if not self.path(s, condition).exists()]
        if not missing: return
        if self.detector is None:
            from perception.winclip_patch import WinCLIPPatch
            self.detector = WinCLIPPatch(device="cpu")
        self.cache.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(missing), 2):
            batch = missing[start:start+2]
            images = []
            for sample in batch:
                if file_hash(ROOT / sample["path"]) != sample["file_sha256"]: raise ValueError("Frozen source file changed")
                with Image.open(ROOT / sample["path"]) as im: images.append(transform(im, condition))
            patches, vectors = self.detector.extract_batch(images)
            for sample, local, vector in zip(batch, patches, vectors):
                path = self.path(sample, condition)
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream: np.savez(stream, patches=local, vector=vector)
                temporary.replace(path)
            if start % 40 == 0: print(json.dumps({"condition": condition, "extracted": start+len(batch), "missing": len(missing)}), flush=True)

    def get(self, sample, condition="original"):
        with np.load(self.path(sample, condition), allow_pickle=False) as z: return z["patches"].copy(), z["vector"].copy()


def threshold(labels, scores, groups):
    labels, scores, groups = np.asarray(labels), np.asarray(scores), np.asarray(groups)
    candidates = np.unique(np.r_[0., scores, np.nextafter(float(scores.max()), np.inf), 1.])
    best = None
    for value in candidates:
        balanced = []
        for group in np.unique(groups):
            keep = groups == group
            y, s = labels[keep], scores[keep]
            if set(y.tolist()) != {0, 1}: raise ValueError("Calibration must contain both labels per group")
            balanced.append(.5 * (np.mean(s[y == 1] >= value) + np.mean(s[y == 0] < value)))
        objective = (float(np.mean(balanced)), -abs(float(value)-.5))
        if best is None or objective > best[0]: best = objective, float(value)
    return best[1]


def fit(output):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    if (output/'report.json').exists():raise ValueError('Completed experiment is frozen; use a new directory for a new fit')
    protocol = read(output / "protocol.json")
    samples = protocol["samples"]
    development = [s for s in samples if s["role"] not in ("test","new_test")]
    cache = Features(output, protocol)
    for condition in set(protocol.get('training_conditions',['original'])+protocol.get('calibration_conditions',['original'])):
        cache.ensure(development,condition)
    training_base = [s for s in development if s["role"] in ("train", "bank")]
    calibration_base = [s for s in development if s["role"] == "calibration"]
    training = [{**s,'condition':c} for s in training_base for c in protocol.get('training_conditions',['original'])]
    calibration = [{**s,'condition':c} for s in calibration_base for c in protocol.get('calibration_conditions',['original'])]
    x = np.array([cache.get(s,s['condition'])[1] for s in training], dtype=np.float64)
    y = np.array([s["label"] for s in training]); g = np.array([s["group"] for s in training])
    xc = np.array([cache.get(s,s['condition'])[1] for s in calibration], dtype=np.float64)
    yc = np.array([s["label"] for s in calibration]); gc = np.array([s["group"] for s in calibration])
    def classifier(features, labels, groups, prefix):
        weights = np.zeros(len(labels))
        for group in np.unique(groups):
            for label in (0, 1):
                keep = (groups == group) & (labels == label)
                if not keep.any(): raise ValueError("Training needs both labels in each group")
                weights[keep] = len(labels) / (2 * len(np.unique(groups)) * keep.sum())
        model = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=.1, max_iter=2000, random_state=20261106))])
        with threadpool_limits(limits=2): model.fit(features, labels, clf__sample_weight=weights)
        params = export_binary(model, prefix)
        if not np.allclose(binary_probability(features[:8], params, prefix), model.predict_proba(features[:8])[:,1], atol=1e-10):
            raise ValueError("Portable classifier parity failed")
        return params
    params = {}
    for group in range(2): params.update(classifier(x[g==group], y[g==group], g[g==group], "separate_"+str(group)))
    params.update(classifier(x, y, g, "pooled"))
    params.update(classifier(conditional_features(x, g, 2), y, g, "structured"))
    separate_thresholds = [threshold(yc[gc==i], binary_probability(xc[gc==i],params,"separate_"+str(i)),gc[gc==i]) for i in range(2)]
    pooled_threshold = threshold(yc, binary_probability(xc,params,"pooled"),gc)
    structured_scores = binary_probability(conditional_features(xc,gc,2),params,"structured")
    structured_thresholds = [threshold(yc[gc==i],structured_scores[gc==i],gc[gc==i]) for i in range(2)]
    banks, centroids, cutoffs = [], [], []
    for group in range(2):
        normal = [s for s in development if s["role"] == "bank" and s["group"] == group]
        patches = np.concatenate([cache.get(s,c)[0] for s in normal for c in protocol.get('training_conditions',['original'])])
        bank = representative_bank(patches, size=protocol["coreset_per_group"],seed=20261106+group)
        banks.append(bank)
        centroid = unit(np.mean([cache.get(s,c)[1] for s in normal for c in protocol.get('training_conditions',['original'])],axis=0));centroids.append(centroid)
        cs = unit(xc[(gc==group)&(yc==0)]) @ centroid
        cutoffs.append(float(max(-1,np.quantile(cs,.01)-.03)))
    params.update(bank=np.concatenate(banks),bank_group=np.repeat(np.arange(2),[len(b) for b in banks]),centroids=np.array(centroids))
    np.savez_compressed(output/"family.npz",**params)
    metadata = {"version":protocol['version'],"groups":protocol["groups"],"arrays":"family.npz",
        "thresholds":{"separate":separate_thresholds,"pooled":pooled_threshold,"structured":structured_thresholds},
        "scope_min_similarity":cutoffs,"protocol_sha256":file_hash(output/"protocol.json"),
        "arrays_sha256":file_hash(output/"family.npz"),"automatic_archival":False,
        "weights_sha256":protocol['source_artifacts']['weights'],
        "training_images":len(training_base),"training_views":len(training),"calibration_images":len(calibration_base),"calibration_views":len(calibration),"bank_images":160,
        "scope":protocol["scope"]}
    write(output/"family.json",metadata)
    model = FamilyModel(output/"family.json")
    raw_cal = []
    for sample in calibration:
        patches,vector=cache.get(sample,sample['condition']);raw=model.patch_scores(patches);route=model.route(vector)
        raw_cal.append({"group":sample["group"],"label":sample["label"],"route":route["group"],
                        "grouped":[float(a.max()) for a in raw["grouped"]],"pooled":float(raw["pooled"].max())})
    memory_thresholds = {"separate":[float(np.quantile([r['grouped'][i] for r in raw_cal if r['group']==i and r['label']==0],.99)) for i in range(2)],
        "pooled":float(np.quantile([r['pooled'] for r in raw_cal if r['label']==0],.99))}
    metadata["memory_thresholds"]=memory_thresholds
    write(output/"family.json",metadata)
    write(output/"calibration_scores.json",raw_cal)
    print(json.dumps({"fitted":True,"training":len(training),"calibration":len(calibration),"thresholds":metadata['thresholds']}),flush=True)


def score_metrics(rows, arm, memory=False):
    y=np.array([r['label'] for r in rows]);s=np.array([r['memory_scores' if memory else 'scores'][arm] for r in rows]);
    cuts=np.array([r['memory_thresholds' if memory else 'thresholds'][arm] for r in rows]);pred=s>=cuts
    tn=int(np.sum((y==0)&~pred));fp=int(np.sum((y==0)&pred));fn=int(np.sum((y==1)&~pred));tp=int(np.sum((y==1)&pred))
    return {"n":len(rows),"normal":tn+fp,"defect":tp+fn,"accuracy":(tn+tp)/len(rows),
        "recall":tp/(tp+fn) if tp+fn else None,"false_positive_rate":fp/(tn+fp) if tn+fp else None,
        "false_positives":fp,"false_negatives":fn,"confusion_matrix":[[tn,fp],[fn,tp]],
        "in_scope":sum(r['in_scope'] for r in rows) if arm=='structured' else len(rows)}


def evaluate(output):
    protocol=read(output/'protocol.json');model=FamilyModel(output/'family.json');cache=Features(output,protocol)
    if model.meta['protocol_sha256']!=file_hash(output/'protocol.json') or model.meta['arrays_sha256']!=file_hash(output/'family.npz'):
        raise ValueError('Frozen model or protocol changed')
    testing=[s for s in protocol['samples'] if s['role']=='test'];all_rows=[]
    for condition in CONDITIONS:
        cache.ensure(testing,condition)
        target=output/('predictions_'+condition+'.json')
        rows=[]
        for n,sample in enumerate(testing):
            patches,vector=cache.get(sample,condition);prediction=model.predict(vector,known_group=sample['group']);raw=model.patch_scores(patches);group=prediction['routing']['group']
            memory_scores={'separate':float(raw['grouped'][sample['group']].max()),'pooled':float(raw['pooled'].max()),'structured':float(raw['grouped'][group].max())}
            memory_cuts={'separate':model.meta['memory_thresholds']['separate'][sample['group']],
                         'pooled':model.meta['memory_thresholds']['pooled'],'structured':model.meta['memory_thresholds']['separate'][group]}
            rows.append({'pixel_hash':sample['pixel_hash'],'group':sample['group'],'label':sample['label'],'condition':condition,
                'scores':prediction['scores'],'thresholds':prediction['thresholds'],'routed_group':group,'in_scope':prediction['routing']['in_scope'],
                'memory_scores':memory_scores,'memory_thresholds':memory_cuts,
                'reference_bank_scores':[float(a.max()) for a in raw['grouped']],
                'reference_similarities':(unit(vector)@model.centroids.T).tolist()})
            if n%80==0:print(json.dumps({'scoring':condition,'completed':n+1,'total':len(testing)}),flush=True)
        write(target,rows);all_rows.extend(rows)
    metrics={condition:{arm:{'by_group':{protocol['groups'][i]['key']:score_metrics([r for r in all_rows if r['condition']==condition and r['group']==i],arm) for i in range(2)},
                'pooled':score_metrics([r for r in all_rows if r['condition']==condition],arm)} for arm in ARMS} for condition in CONDITIONS}
    for condition in metrics:
        for arm in ARMS:
            metrics[condition][arm]['macro_accuracy']=float(np.mean([m['accuracy'] for m in metrics[condition][arm]['by_group'].values()]))
            metrics[condition][arm]['macro_recall']=float(np.mean([m['recall'] for m in metrics[condition][arm]['by_group'].values()]))
    memory={condition:{arm:{protocol['groups'][i]['key']:score_metrics([r for r in all_rows if r['condition']==condition and r['group']==i],arm,True) for i in range(2)} for arm in ARMS} for condition in CONDITIONS}
    reasons=[];base=metrics['original']['separate'];candidate=metrics['original']['structured']
    if candidate['macro_accuracy']<base['macro_accuracy']-.02:reasons.append('原图跨来源平均准确率较分别建模下降超过2个百分点')
    for group in protocol['groups']:
        key=group['key']
        if candidate['by_group'][key]['recall']<base['by_group'][key]['recall']-.02:reasons.append(key+'原图缺陷召回下降超过2个百分点')
    routing={condition:{'n':len(testing),'source_routing_errors':sum(r['routed_group']!=r['group'] for r in all_rows if r['condition']==condition),
              'out_of_scope':sum(not r['in_scope'] for r in all_rows if r['condition']==condition)} for condition in CONDITIONS}
    transfer=[]
    for reference in range(2):
        rows=[dict(r) for r in all_rows if r['condition']=='original' and r['group']!=reference]
        for r in rows:
            r['memory_scores']={'separate':r['reference_bank_scores'][reference]}
            r['memory_thresholds']={'separate':model.meta['memory_thresholds']['separate'][reference]}
        transfer.append({'reference':protocol['groups'][reference]['key'], 'target':protocol['groups'][1-reference]['key'],
            'normal_only_forced_classification':score_metrics(rows,'separate',True),
            'scope_rejected':sum(r['reference_similarities'][reference]<model.meta['scope_min_similarity'][reference] for r in rows),
            'scope':'diagnostic cross-source transfer; target images absent from this reference/calibration; no threshold tuning'})
    report={'version':protocol['version'],'groups':protocol['groups'],'conditions':CONDITIONS,'base_test_images':len(testing),
        'metrics':metrics,'normal_only_memory':memory,'routing':routing,'cross_source_transfer_diagnostic':transfer,'eligible_for_advisory_pilot':not reasons,
        'gate_reasons':reasons,'automatic_archival':False,'label_budget':{'training':model.meta['training_images'],'calibration':model.meta['calibration_images'],'bank_normal':160},
        'scope':protocol['scope'],'protocol_sha256':file_hash(output/'protocol.json'),'arrays_sha256':file_hash(output/'family.npz'),
        'caution':'in_scope is an appearance screen; no SKU identification or guaranteed unseen-domain performance; stress views are correlated versions of the same images'}
    write(output/'report.json',report)
    print(json.dumps({'finished':True,'pilot_eligible':not reasons,'routing':routing,'original_macro':{a:metrics['original'][a]['macro_accuracy'] for a in ARMS}}),flush=True)


def summarize(output):
    report=read(output/'report.json');protocol=read(output/'protocol.json')
    if report['protocol_sha256']!=file_hash(output/'protocol.json'):raise ValueError('Report and protocol differ')
    names={'separate':'分别建模（已知来源/视图）','pooled':'完全合并','structured':'结构化统一库（自动匹配）'}
    conditions={'original':'原图回归','dim':'模拟变暗 ×0.65','warm':'模拟偏暖 RGB×(1.12,1,0.82)'}
    pc=lambda n:'—' if n is None else f'{n*100:.2f}%'
    lines=['# 螺丝产品族统一库对照实验','', '日期：2026-11-06。网站：https://metalqc.online/ 。入口：产品接入 → 螺丝产品族·统一库验证。','',
        '## 实验回答的问题','',
        '同类零件能否共用模型和正常参考库，并自动匹配不同来源、视图？本轮比较分别处理、完全混合、结构化统一三种方案，不预设合并一定更好。','',
        '本轮只有 MVTec 螺丝视图和 M2 头部俯视两个来源。数据没有明确、完整的型号和实物身份信息，因此结果是跨数据源/视图验证，不能称为全部型号泛化或新产线盲测。','',
        '## 固定划分与公平性','',
        '| 来源 | 正常参考 | 原有标注训练 | 校准 | 留出回归 |','| --- | ---: | ---: | ---: | ---: |']
    for i,group in enumerate(report['groups']):
        counts=Counter(s['role'] for s in protocol['samples'] if s['group']==i)
        lines.append(f"| {group['name']} | {counts['bank']} | {counts['train']} | {counts['calibration']} | {counts['test']} |")
    lines+=['',
        '整图判别训练共640张，包含两组正常参考图160张（正常标签可用于拟合）和原有标注训练图480张。三方案使用同一冻结CLIP主干、同一批训练图片、同一校准分区；超参数事先固定。合并训练对来源及正常/缺陷做等权平衡，避免大来源压过小来源。','',
        'MVTec官方test图此前已按训练80、校准40、留出40分区，另有40张正常训练图用于校准；本轮只把留出40张计作评估。M2保留发布方536张Test。两组均已用于历史实验，本轮称回归，不称新增盲测。','',
        '校准集决定阈值，测试图没有参与拟合、阈值选择或模型调参。每个原始测试图产生原图、模拟变暗、模拟偏暖三种版本；共有576张独立于本次拟合的原始图，不能把1728个相关版本当作1728张独立新样本。','',
        '## 三种方案分别是什么','',
        '| 方案 | 整图判断 | 正常参考检索 | 输入时需要选择来源吗 |','| --- | --- | --- | --- |',
        '| 分别处理 | 每来源一个判别头 | 使用对应参考子集 | 是；作为已知来源对照 |',
        '| 完全合并 | 一个共用判别头、一个阈值 | 所有正常特征直接混合 | 否 |',
        '| 结构化统一 | 一个共用条件判别头，输入自动匹配的参考组 | 一个总库保留来源/视图标签，按自动匹配组检索 | 否；从图片特征匹配 |','',
        '每组80张正常图选1024个代表局部特征；总存储预算均为2048个特征。结构化方案不是每个型号维护一套完整大模型；其参考组目前代表两个来源/视图，不是型号识别器。各组可使用不同校准阈值，共享模型并不要求所有分布使用相同分数边界。','',
        '## 有标签整图判别结果','',
        '下表的准确率和召回率按两个来源等权平均。它们是来源标签下的正常/缺陷判定，包含缺陷标签训练，不能归因于正常库或称为零样本效果。','',
        '| 条件 | 方案 | 两来源平均准确率 | 两来源平均缺陷召回 |','| --- | --- | ---: | ---: |']
    for c in conditions:
        for a in ARMS:
            m=report['metrics'][c][a];lines.append(f"| {conditions[c]} | {names[a]} | {pc(m['macro_accuracy'])} | {pc(m['macro_recall'])} |")
    lines+=['','### 分来源错误数','','| 条件 | 方案 | 来源 | 张数 | 误报 | 漏检 | 准确率 | 缺陷召回 |','| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for c in conditions:
        for a in ARMS:
            for group in report['groups']:
                m=report['metrics'][c][a]['by_group'][group['key']]
                lines.append(f"| {conditions[c]} | {names[a]} | {group['name']} | {m['n']} | {m['false_positives']} | {m['false_negatives']} | {pc(m['accuracy'])} | {pc(m['recall'])} |")
    if report.get('routed_expert_replay'):
        lines+=['','## 补充：自动匹配分组头的工程回放','',
            '观察到共用判别头的原图漏检增加后，补充回放一种工程方案：沿用已经冻结的两个轻量判别头，由同一个视觉编码器自动匹配来源/视图，再调用对应头。正常参考仍存放在一个总库中。没有重新拟合、没有重调阈值；该补充为后验分析，不是预注册第四臂或新增盲测。','',
            '| 条件 | 来源 | 准确率 | 缺陷召回 | 误报 | 漏检 |','| --- | --- | ---: | ---: | ---: | ---: |']
        for c in conditions:
            for group in report['groups']:
                m=report['routed_expert_replay']['metrics'][c][group['key']]
                lines.append(f"| {conditions[c]} | {group['name']} | {pc(m['accuracy'])} | {pc(m['recall'])} | {m['false_positives']} | {m['false_negatives']} |")
        lines+=['','该方案统一了模型主干、参考库和入口，保留少量分组判别头，不能称为完全共用一个判别头。参考组可覆盖多种相近外观；目前只有两个来源/视图，不能据此断言任意新型号都适用。网站试检优先展示该工程回放结果，原三臂的失败结果完整保留。']
    lines+=['','## 正常库单独评分','','只使用正常参考，不训练缺陷判别头。阈值取正常校准分数99百分位；结构化方案按自动匹配参考使用相应边界。此表用于直接观察混合记忆库的影响。','','| 条件 | 方案 | 来源 | 误报 | 漏检 | 准确率 | 缺陷召回 |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for c in conditions:
        for a in ARMS:
            for group in report['groups']:
                m=report['normal_only_memory'][c][a][group['key']]
                lines.append(f"| {conditions[c]} | {names[a]} | {group['name']} | {m['false_positives']} | {m['false_negatives']} | {pc(m['accuracy'])} | {pc(m['recall'])} |")
    lines+=['','## 自动参考匹配与未覆盖来源','','| 条件 | 参考组匹配错误 | 初筛范围外 | 总数 |','| --- | ---: | ---: | ---: |']
    for c in conditions:
        r=report['routing'][c];lines.append(f"| {conditions[c]} | {r['source_routing_errors']} | {r['out_of_scope']} | {r['n']} |")
    lines+=['','匹配正确说明能区分本轮两个来源/视图，不能证明认出了具体型号。初筛通过也不证明质量判定正确。未覆盖来源的诊断只使用另一来源的正常参考和阈值，不用目标来源校准：','','| 可用参考 | 测试来源 | 正常误报 | 缺陷漏检 | 初筛范围外 |','| --- | --- | ---: | ---: | ---: |']
    for r in report['cross_source_transfer_diagnostic']:
        m=r['normal_only_forced_classification'];lines.append(f"| {r['reference']} | {r['target']} | {m['false_positives']}/{m['normal']} | {m['false_negatives']}/{m['defect']} | {r['scope_rejected']}/{m['n']} |")
    lines+=['','## 本轮决策','',
        ('结构化统一方案通过事先设定的原图不回退检查，可用于有明确范围的辅助试检。' if report['eligible_for_advisory_pilot'] else '结构化统一方案未通过事先设定的原图不回退检查，保留为研究对照，不能替换正式检测配置。'),'']
    lines += ['- '+r for r in report['gate_reasons']]
    lines+=['',
        '门槛为：原图每个来源的缺陷召回相对分别建模下降不超过2个百分点，来源平均准确率下降不超过2个百分点；它只检查本次回归，不构成自动放行资格。模拟光照结果用于暴露适用边界，没有据此重新选阈值。','',
        '网站提供一个螺丝族试检入口，用户上传图片后自动匹配参考，不必先选择这两个来源。当前试检建议使用后验工程回放的自动分组头结果，页面同时保留共用判别头对照、模型阈值和正常参考差异热力图；试检不调用VLM、不归档、不生成反馈训练标签、不改动原有四个产品配置。正式自动归档保持原策略，M2旧版本的质量暂停不被此次实验解除。','',
        '下一阶段应取得带明确型号、采集条件及实物/批次编号的多型号数据，按型号与采集批次整组留出，分别评估已覆盖型号换环境、完全未覆盖型号两种情况。对明显光照敏感的问题，先改进采集、目标对齐或预先定义的数据增强，再在未参与开发的新数据上验证。','',
        '## 复现与证据','',
        '实验目录：`results/experiments/screw_family_v1/`。`protocol.json`固定图片分区、文件/像素哈希、主干哈希及超参数；`family.npz`为统一模型与带标签参考总库；`family.json`为阈值和哈希；`predictions_*.json`为逐图结果；`report.json`为汇总。','',
        '```powershell',"$env:PYTHONPATH = 'src'",'D:\\Anaconda3\\python.exe src/experiments/screw_family.py prepare','D:\\Anaconda3\\python.exe src/experiments/screw_family.py fit','D:\\Anaconda3\\python.exe src/experiments/screw_family.py evaluate','D:\\Anaconda3\\python.exe src/experiments/screw_family.py summarize','```','',
        '完成的实验禁止重新拟合覆盖；新方案使用新的输出目录。预训练主干、模型数组与报告的哈希在推理前核验。数据来源：'+ '；'.join(g['source'] for g in report['groups'])+'。']
    target=ROOT/'docs/螺丝统一库对照实验_20261106.md';target.write_text('\n'.join(lines)+'\n',encoding='utf8');print(str(target),flush=True)


def expert_replay(output):
    """Post-hoc engineering replay; no fit, threshold changes or new-test claim."""
    report=read(output/'report.json');protocol=read(output/'protocol.json');model=FamilyModel(output/'family.json');cache=Features(output,protocol)
    index={s['pixel_hash']:s for s in protocol['samples'] if s['role']=='test'}
    metrics={};evidence=[]
    for condition in CONDITIONS:
        rows=read(output/('predictions_'+condition+'.json'));replayed=[]
        for row in rows:
            _,vector=cache.get(index[row['pixel_hash']],condition)
            result=model.predict(vector)
            if result['routing']['group']!=row['routed_group']:raise ValueError('Frozen reference routing changed')
            updated={**row,'scores':{**row['scores'],'routed_expert':result['scores']['routed_expert']},
                     'thresholds':{**row['thresholds'],'routed_expert':result['thresholds']['routed_expert']}}
            replayed.append(updated)
            evidence.append({'pixel_hash':row['pixel_hash'],'condition':condition,'routed_group':result['routing']['group'],
                             'score':result['scores']['routed_expert'],'threshold':result['thresholds']['routed_expert']})
        metrics[condition]={group['key']:score_metrics([r for r in replayed if r['group']==i],'routed_expert') for i,group in enumerate(protocol['groups'])}
    report['routed_expert_replay']={'method':('predeclared routed-head evaluation' if protocol.get('expert_replay_prespecified') else 'post-hoc replay, shared backbone and tagged unified bank with routed existing lightweight heads'),
        'prespecified':bool(protocol.get('expert_replay_prespecified')),
        'trained_on_test':False,'thresholds_changed':False,'fresh_blind_test':False,'metrics':metrics,'automatic_archival':False}
    write(output/'routed_expert_predictions.json',evidence);write(output/'report.json',report)
    print(json.dumps({'routed_expert_replay':True,'original':metrics['original']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','fit','evaluate','expert-replay','summarize'));parser.add_argument('--output',type=Path,default=ROOT/'results/experiments/screw_family_v1');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(2)
    {'prepare':prepare,'fit':fit,'evaluate':evaluate,'expert-replay':expert_replay,'summarize':summarize}[args.mode](args.output)


if __name__=='__main__':main()
