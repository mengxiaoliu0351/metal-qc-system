"""V2: declared illumination training; new retained images frozen before fitting."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from screw_family import Features,CONDITIONS,FamilyModel,M2,BUILD,fit,evaluate,expert_replay,score_metrics
from review.workflow import read,write,file_hash,image_hash

V1=ROOT/'results/experiments/screw_family_v1'
V2=ROOT/'results/experiments/screw_family_v2'


def prepare():
    previous=read(V1/'protocol.json')
    protocol=json.loads(json.dumps(previous))
    protocol.update(version='screw-family-v2-illumination',training_conditions=list(CONDITIONS),calibration_conditions=list(CONDITIONS),
        feature_cache='results/experiments/screw_family_v1/features',
        development_context='V1 synthetic-light failures observed before choosing this augmentation; old 576-image split is development regression',
        fresh_protocol='300 previously unused retained images; image-level only; 100 MVTec normals and 100 M2 normals + 100 M2 defects; no new MVTec defect recall claim',
        fresh_gate={'minimum_m2_recall_each_condition':.9,'maximum_m2_fpr_each_condition':.1,'maximum_mvtec_normal_fpr_each_condition':.1},
        expert_replay_prespecified=True)
    existing={s['pixel_hash'] for s in protocol['samples']}
    replayed=read(ROOT/'results/experiments/engineering_acceptance_20261105/cloud_feedback_batch.json')
    existing.update(i['id'] for i in replayed['items'])
    m2=read(ROOT/f'results/product_catalog/{M2}/builds/{BUILD}/input.json')
    additions=[]
    for label in (0,1):
        candidates=sorted((s for s in m2['samples'] if s['role']=='feedback' and (s['label']=='defect')==bool(label)
                           and s['pixel_hash'] not in existing),key=lambda s:s['pixel_hash'])
        if len(candidates)<100:raise ValueError('Need 100 unused M2 images per label')
        for s in candidates[:100]:
            additions.append({'path':s['path'],'pixel_hash':s['pixel_hash'],'file_sha256':file_hash(ROOT/s['path']),
                              'group':1,'role':'new_test','label':label})
    legacy=read(ROOT/'src/review/calibration_exclusions.json')['screw']['legacy_memory']
    existing.update(image_hash(ROOT/p) for p in legacy)
    normals=[]
    for path in sorted((ROOT/'data/mvtec/screw/train/good').glob('*.png')):
        fingerprint=image_hash(path)
        if fingerprint not in existing:
            normals.append({'path':path.relative_to(ROOT).as_posix(),'pixel_hash':fingerprint,'file_sha256':file_hash(path),
                            'group':0,'role':'new_test','label':0})
    normals.sort(key=lambda s:s['pixel_hash'])
    if len(normals)<100:raise ValueError('Need 100 previously unused MVTec normal images')
    additions+=normals[:100]
    if len({s['pixel_hash'] for s in additions})!=300:raise ValueError('Fresh images duplicated')
    protocol['samples']+=additions
    target=V2/'protocol.json'
    if target.exists() and read(target)!=protocol:raise ValueError('Frozen V2 protocol differs')
    write(target,protocol)
    reservation_path=ROOT/'results/evaluation_reservations.json'
    reservations=read(reservation_path) if reservation_path.exists() else {'version':1,'products':{}}
    for sample in additions:
        product='screw' if sample['group']==0 else M2
        reservations['products'].setdefault(product,{})[sample['pixel_hash']]={'experiment':protocol['version'],'role':'new_test'}
    write(reservation_path,reservations)
    print(json.dumps({'v2_frozen':True,'fresh_images':300,'conditions':list(CONDITIONS),'counts':{str(k):v for k,v in Counter((s['group'],s['label']) for s in additions).items()}}),flush=True)


def fresh():
    protocol=read(V2/'protocol.json');model=FamilyModel(V2/'family.json');baseline=FamilyModel(V1/'family.json');cache=Features(V2,protocol)
    samples=[s for s in protocol['samples'] if s['role']=='new_test'];rows=[]
    for condition in CONDITIONS:
        cache.ensure(samples,condition)
        for sample in samples:
            _,vector=cache.get(sample,condition)
            for version,m in [('v1',baseline),('v2',model)]:
                result=m.predict(vector)
                rows.append({'pixel_hash':sample['pixel_hash'],'condition':condition,'group':sample['group'],'label':sample['label'],
                    'version':version,'scores':result['scores'],'thresholds':result['thresholds'],'in_scope':result['routing']['in_scope'],
                    'routed_group':result['routing']['group']})
        print(json.dumps({'fresh_condition_complete':condition,'images':len(samples)}),flush=True)
    metrics={version:{c:{group['key']:score_metrics([r for r in rows if r['version']==version and r['condition']==c and r['group']==i],'routed_expert')
             for i,group in enumerate(protocol['groups'])} for c in CONDITIONS} for version in ('v1','v2')}
    all_metrics={version:{arm:{c:{group['key']:score_metrics([r for r in rows if r['version']==version and r['condition']==c and r['group']==i],arm)
                 for i,group in enumerate(protocol['groups'])} for c in CONDITIONS} for arm in ('pooled','structured','routed_expert')} for version in ('v1','v2')}
    routing={version:{c:{'source_routing_errors':sum(r['routed_group']!=r['group'] for r in rows if r['version']==version and r['condition']==c),
                        'out_of_scope':sum(not r['in_scope'] for r in rows if r['version']==version and r['condition']==c)} for c in CONDITIONS} for version in ('v1','v2')}
    reasons=[]
    for c in CONDITIONS:
        m2=metrics['v2'][c]['m2_head'];mv=metrics['v2'][c]['mvtec_screw']
        if m2['recall']<.9:reasons.append(c+': M2新增留出缺陷召回低于90%')
        if m2['false_positive_rate']>.1:reasons.append(c+': M2新增留出正常误报率高于10%')
        if mv['false_positive_rate']>.1:reasons.append(c+': MVTec新增正常误报率高于10%')
    result={'base_images':len(samples),'condition_views':len(samples)*len(CONDITIONS),'metrics':metrics,'all_methods':all_metrics,'routing':routing,
        'gate_reasons':reasons,'fresh_gate_passed':not reasons,'automatic_archival':False,
        'scope':protocol['fresh_protocol'],'development_context':protocol['development_context'],
        'protocol_sha256':file_hash(V2/'protocol.json'),'arrays_sha256':file_hash(V2/'family.npz')}
    write(V2/'fresh_predictions.json',rows);write(V2/'fresh_report.json',result)
    report=read(V2/'report.json');report['new_retained_evaluation']=result
    report['prior_experiment']={'version':'screw-family-v1','scope':'original-only training; preserved failed common-head experiment',
        'metrics':read(V1/'report.json')['metrics'],'routing':read(V1/'report.json')['routing']}
    write(V2/'report.json',report)
    print(json.dumps({'fresh_gate_passed':not reasons,'reasons':reasons,'v2_metrics':metrics['v2']}),flush=True)


def summarize():
    first=read(V1/'report.json');report=read(V2/'report.json');fresh_result=read(V2/'fresh_report.json');meta=read(V2/'family.json')
    pc=lambda v:'—' if v is None else f'{v*100:.2f}%'
    conditions={'original':'原图','dim':'模拟变暗（亮度×0.65）','warm':'模拟偏暖（RGB×1.12/1/0.82）'}
    groups=report['groups']
    lines=['# 螺丝统一库与多光照改进：两轮实验及新增留出验证','',
        '更新日期：2026-11-06。公网 https://metalqc.online/ → 产品接入 → 螺丝产品族·统一库验证。','',
        '## 当前交付','',
        '一个螺丝族试检入口，共享一个CLIP视觉编码器和一个正常参考总库；按图片自动匹配来源/视图参考组，再使用轻量分组判别头。用户不需要先选择这两个数据来源。保留两个轻量头，不宣称完全共用一个判别头；参考分组代表来源/视图，并非一个型号一个库。','',
        '已验证的范围只有MVTec螺丝视图和M2头部俯视。两个来源不能代表全部螺丝型号、厂家和实际光照环境。研究试检不自动归档、不生成训练标签、不修改原有四个产品配置。','',
        '## 第一轮为什么没有直接上线共用判别头','',
        '固定同样的训练、校准和留出图片，比较分别处理、完全合并、结构化统一三个方案。原图结果如下：','',
        '| 方案 | 两来源平均准确率 | 两来源平均缺陷召回 | M2误报 | M2漏检 |','| --- | ---: | ---: | ---: | ---: |']
    for arm,name in [('separate','分别处理，已知来源'),('pooled','完全合并'),('structured','共用条件判别头＋自动参考匹配')]:
        m=first['metrics']['original'][arm];v=m['by_group']['m2_head'];lines.append(f"| {name} | {pc(m['macro_accuracy'])} | {pc(m['macro_recall'])} | {v['false_positives']} | {v['false_negatives']} |")
    lines+=['','共用条件判别头减少了M2正常误报，却增加了缺陷漏检，因此没有将它替换为正式判定。将不同分布的数据混合，并不自动保证效果更好。第一轮也暴露了光照敏感性。完整三臂及正常库单独评分保存在《螺丝统一库对照实验_20261106.md》和`screw_family_v1`目录，没有删除失败结果。','',
        '## 第二轮做了什么','',
        '- 保持原始训练图片和分区不变，640张原始训练图生成原图、变暗、偏暖共1920个训练视图；280张校准图对应840个校准视图。视图不是新增独立样本。',
        '- 每来源仍用80张正常参考图，加入相同光照变化；压缩后每来源1024个代表局部特征，一个总库共2048个特征。没有按每个型号复制大模型。',
        '- 保留自动参考匹配和轻量分组头；同一编码器一次提取图像特征，不调用VLM。',
        '- 在训练前固定300张新增留出图片，并按像素去重排除本轮训练、校准、参考和之前40张标签回放图片。',
        '- 第一轮光照问题已经被观察过，因此原有576张只作为开发回归；改进不包装成对这576张的新盲测。','',
        '| 新增留出来源 | 正常 | 缺陷 | 能检验什么 |','| --- | ---: | ---: | --- |',
        '| MVTec原正常训练池中未用于本轮建模及旧参考的图片 | 100 | 0 | 误报，不能测新增缺陷召回 |',
        '| M2预留反馈池，排除已回放图片 | 100 | 100 | 正常误报与缺陷漏检 |','',
        '来源没有实物身份编号。去重只证明像素内容不同，不能证明来自不同实物或独立产线。新留出也来自现有公开数据，不能替代真实采集条件验证。','',
        '## 原有576张开发回归：自动匹配分组头','',
        '| 条件 | 来源 | 原图训练准确率 | 多光照训练准确率 | 原图训练误报/漏检 | 多光照训练误报/漏检 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for c in conditions:
        for group in groups:
            old=first['routed_expert_replay']['metrics'][c][group['key']];new=report['routed_expert_replay']['metrics'][c][group['key']]
            lines.append(f"| {conditions[c]} | {group['name']} | {pc(old['accuracy'])} | {pc(new['accuracy'])} | {old['false_positives']}/{old['false_negatives']} | {new['false_positives']}/{new['false_negatives']} |")
    lines+=['','## 新增300张留出：训练完成后实际评价','',
        '以下两版在相同300张图片及其模拟视图上比较。每条件300张，三条件共900个相关视图，不能计作900张独立新样本。MVTec只有正常图，召回率显示“—”。','',
        '| 条件 | 来源 | 版本 | 准确率 | 缺陷召回 | 正常误报率 | 误报/漏检 |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for c in conditions:
        for group in groups:
            for version,name in [('v1','原图训练'),('v2','多光照训练')]:
                m=fresh_result['metrics'][version][c][group['key']]
                lines.append(f"| {conditions[c]} | {group['name']} | {name} | {pc(m['accuracy'])} | {pc(m['recall'])} | {pc(m['false_positive_rate'])} | {m['false_positives']}/{m['false_negatives']} |")
    lines+=['','### 一个共用判别头的新增留出对照','','它与自动分组头使用相同训练/校准数据；下表保留其结果，避免把“统一参考库”和“完全共用判别头”混为一谈。','','| 条件 | 来源 | 共用条件判别头准确率 | 共用条件判别头缺陷召回 | 自动分组头准确率 | 自动分组头缺陷召回 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for c in conditions:
        for group in groups:
            shared=fresh_result['all_methods']['v2']['structured'][c][group['key']];expert=fresh_result['metrics']['v2'][c][group['key']]
            lines.append(f"| {conditions[c]} | {group['name']} | {pc(shared['accuracy'])} | {pc(shared['recall'])} | {pc(expert['accuracy'])} | {pc(expert['recall'])} |")
    lines+=['','## 门槛与当前边界','',
        '新增评估前固定辅助试检检查门槛：每种模拟条件下，M2缺陷召回至少90%，M2正常误报不高于10%，新增MVTec正常误报不高于10%。这些是本轮研究检查门槛，不能视为工业验收或自动放行标准。','',
        ('新增留出检查全部通过。' if fresh_result['fresh_gate_passed'] else '新增留出检查未全部通过，具体原因：')]
    lines+=['- '+r for r in fresh_result['gate_reasons']]
    lines+=['',
        '无论本轮检查是否通过，自动归档均为关闭，原有M2版本的质量暂停不会自动解除。完全共用条件判别头在原图不回退检查的结果仍单独保留：'+ ('通过。' if report['eligible_for_advisory_pilot'] else '未通过：'+'；'.join(report['gate_reasons'])+'。'),'',
        '多光照训练针对已发现的两种模拟变化；尚未证明真实灯具、相机、遮挡、反光、尺寸变化、陌生型号和厂家之间的稳定性。相近型号可尝试纳入同一参考组，但需要正常与缺陷对照验证，不能凭名称相同直接合并。','',
        '## 网站怎么用','',
        '进入“产品接入”，找到“螺丝产品族·统一库验证”，上传图片并点击“使用统一库试检”。系统自动匹配参考，展示单一当前建议及差异热力图；其他试验方案与完整结果折叠展示，避免多个结论同时占据主操作区。','',
        '## 证据与复现','',
        '`results/experiments/screw_family_v1/`保留初始三臂、后验工程回放及失败结果；`results/experiments/screw_family_v2/`保存多光照协议、统一模型、开发回归、新增留出逐图结果。`family_active.json`只选择研究试检版本，不更改正式产品注册表。','',
        '新增留出清单与去重哈希：`screw_family_v2/protocol.json`；新增结果：`fresh_report.json`、`fresh_predictions.json`；模型：`family.npz`与`family.json`。在线推理校验模型数组及视觉主干哈希。','',
        '```powershell',r'$env:PYTHONPATH = "src"',r'D:\Anaconda3\python.exe src/experiments/family_illumination.py prepare',r'D:\Anaconda3\python.exe src/experiments/family_illumination.py fit',r'D:\Anaconda3\python.exe src/experiments/family_illumination.py evaluate',r'D:\Anaconda3\python.exe src/experiments/family_illumination.py fresh',r'D:\Anaconda3\python.exe src/experiments/family_illumination.py summarize','```','',
        '完成目录禁止覆盖重新拟合，改进方案应使用新版本协议与新的评估记录。报告内所有结果均从实验JSON生成，未手改误报、漏检或准确率。']
    target=ROOT/'docs/螺丝统一库与光照改进_20261106.md';target.write_text('\n'.join(lines)+'\n',encoding='utf8');print(str(target))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','fit','evaluate','fresh','summarize'));args=parser.parse_args()
    import torch
    torch.set_num_threads(2)
    V2.mkdir(parents=True,exist_ok=True)
    if args.mode=='prepare':prepare()
    elif args.mode=='fit':fit(V2)
    elif args.mode=='evaluate':evaluate(V2);expert_replay(V2)
    elif args.mode=='fresh':fresh()
    else:summarize()


if __name__=='__main__':main()
