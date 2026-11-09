"""Collect existing screw evidence and runtime model identities; do not fit anything."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from products.families import read, write
from products.screw_delivery import checked_metrics, EVIDENCE
from review.workflow import file_hash

ROOT = Path(__file__).resolve().parents[2]


def head_quality_cohorts(previous, previous_hash, active, active_hash, candidate=None, report=None):
    """Bind each result to its evaluated model, never relabel an old score."""
    old = previous['recipe']['config']; e = previous['evaluation']
    if previous_hash not in previous['recipe']['files'].values():
        raise ValueError('Historical head batch is not paired with its original model')
    tn=e['normal']-e['local_false_positives'];tp=e['defect']-e['local_false_negatives']
    old_current = (active_hash == previous_hash and active['threshold'] == old['threshold']
                   and active['t_high'] == old['t_high'])
    rows = [{'id': 'm2_head' if old_current else 'm2_head_previous', 'view': 'm2_head',
             'name': 'M2头部 · 当前判别配置' if old_current else 'M2头部 · 原评分（历史对照）',
             'population': '官方536张Test历史网站批次',
             'scope': '原评分网站批次的本地二分类；未计参考拒判、VLM或人工纠错，不是新增独立盲测',
             'state': '检测可用，自动处置暂停' if old_current else '历史版本，已被替换',
             'metrics': checked_metrics({'n': e['scored'], 'confusion_matrix': [[tn,e['local_false_positives']],[e['local_false_negatives'],tp]]}),
             'runtime_model_arrays_sha256': previous_hash,
             'runtime_threshold': old['threshold'], 'runtime_t_high': old['t_high']}]
    if old_current:
        return rows
    if not candidate or not report or candidate['arrays_sha256'] != active_hash or candidate['thresholds'] != [active['threshold'],active['t_high']]:
        raise ValueError('Active head scoring has no matching model and threshold evidence')
    metrics = checked_metrics(report['cohorts']['test']['candidate'])
    if any(metrics[k] != rows[0]['metrics'][k] for k in ('n','normal','defect')):
        raise ValueError('Current and historical head populations differ')
    rows.append({'id':'m2_head','view':'m2_head','name':'M2头部 · 当前扩充训练评分',
                 'population':'同一官方536张Test历史回归',
                 'scope':'当前评分的离线本地二分类；未计参考拒判、VLM或人工纠错，非新增完整网站测试或独立盲测',
                 'state':'当前辅助评分，强制人工复核','metrics':metrics,
                 'runtime_model_arrays_sha256':active_hash,
                 'runtime_threshold':active['threshold'],'runtime_t_high':active['t_high']})
    return rows


def build():
    results = ROOT/'results'; sources = {}
    def source(relative):
        path = ROOT/relative; sources[relative] = file_hash(path); return read(path)
    products = {p['key']: p for p in read(results/'perception/banks/products.json')}
    def model_hash(key):
        meta_path = results/'perception/optimized'/products[key]['perception_artifact']
        return file_hash(meta_path.parent/read(meta_path)['arrays'])
    data = []; datasets = []; construction = []
    for name, filename in [('MVTec螺丝近景', 'screw'), ('M2头部', 'head'), ('M2螺杆', 'body')]:
        if filename == 'screw':
            counts = Counter('normal' if p.parent.name == 'good' else 'defect' for split in ('train', 'test')
                             for p in (ROOT/'data/mvtec/screw'/split).glob('*/*') if p.suffix.lower() == '.png')
        else:
            manifest = source('data/external/m2_v2/'+filename+'-manifest.json')
            counts = Counter(s['label'] for s in manifest['images'])
        datasets.append({'name': name, 'images': sum(counts.values()), 'normal': counts['normal'], 'defect': counts['defect'], 'unit': '图像', 'quality_labels': True})
    screw = source('results/experiments/perception_upgrade_v1/screw_report.json')
    split = source('results/experiments/perception_upgrade_v1/screw_split.json')
    construction.append({'view': '螺丝近景', 'normal_reference_images': len(split['normal_memory']),
                         'fitting_images': len(split['train'])+len(split['normal_memory']),
                         'calibration_images': screw['validation'][screw['selected']]['n'], 'evaluation_images': len(split['test']),
                         'note': '正常参考图同时参与整图头拟合；校准计入独立正常图并排除旧库重叠；40张为历史留出。'})
    activation = source('results/experiments/perception_upgrade_v1/activation.json')
    historical_model = next(d['artifact_sha256'] for d in activation['decisions'] if d['product'] == 'screw')
    if historical_model != model_hash('screw'):
        raise ValueError('Historical screw report is not paired with current model')
    data.append({'id': 'mvtec_screw', 'view': 'mvtec_screw', 'name': '螺丝近景 · 当前判别配置',
                 'population': '固定40张历史留出图', 'scope': '本地图像二分类；未计VLM或人工纠错，非新增厂家盲测',
                 'state': '已绑定检测', 'metrics': checked_metrics(screw['test'][screw['selected']]),
                 'runtime_model_arrays_sha256': historical_model,
                 'runtime_threshold': screw['threshold'], 'runtime_t_high': products['screw']['t_high']})
    head_path = 'results/experiments/engineering_acceptance_20261105/e5ea97c485294d519e6ff1b05d7e0fe1_batch.json'
    head = source(head_path)
    head_manifest = source('data/external/m2_v2/head-manifest.json')
    roles = head_manifest['summary']['roles']
    head_key='product_a269405b7b9a';active_head=products[head_key]
    old_config=head['recipe']['config']
    old_meta=source('results/perception/optimized/'+old_config['perception_artifact'])
    old_arrays=results/'perception/optimized'/old_meta['arrays'];old_head_hash=file_hash(old_arrays)
    sources[old_arrays.relative_to(ROOT).as_posix()]=old_head_hash
    current_head_hash=model_hash(head_key)
    expanded = current_head_hash != old_head_hash or any(active_head[k] != old_config[k] for k in ('threshold','t_high'))
    construction.append({'view': 'M2头部（原评分）' if expanded else 'M2头部', 'normal_reference_images': sum(roles['bank'].values()),
                         'fitting_images': sum(roles['train'].values()), 'calibration_images': sum(roles['calibration'].values()),
                         'evaluation_images': sum(roles['test'].values()),
                         'note': '另有200张历史资格检查；原评分已被替换。' if expanded else '另有200张资格检查；检测配置已接入，自动处置暂停。'})
    candidate=report=None
    if expanded:
        directory='results/experiments/m2_head_expanded_v1/'
        candidate=source(directory+'candidate.json');report=source(directory+'report.json');protocol=source(directory+'protocol.json')
        if sources[directory+'protocol.json'] != candidate['protocol_sha256'] or protocol['base_config'] != old_config:
            raise ValueError('Expanded head evidence does not match the frozen baseline protocol')
        metadata=source('results/perception/optimized/'+active_head['perception_artifact'])
        if metadata['threshold'] != active_head['threshold'] or metadata['t_high'] != active_head['t_high']:
            raise ValueError('Current head metadata and scoring thresholds differ')
        construction.append({'view':'M2头部（当前辅助评分）','normal_reference_images':sum(roles['bank'].values()),
                             'fitting_images':sum(s['role']=='train' for s in protocol['samples']),
                             'calibration_images':sum(s['role']=='calibration' for s in protocol['samples']),
                             'evaluation_images':report['cohorts']['test']['candidate']['n'],
                             'note':'原400张训练图加920张非封存来源Train图；正常库不变；仅更新辅助评分，强制人工复核。'})
    data.extend(head_quality_cohorts(head,old_head_hash,active_head,current_head_hash,candidate,report))
    body = source('results/experiments/m2_body_quality_v2/report.json')
    body_protocol = source('results/experiments/m2_body_quality_v2/protocol.json')
    prior_body = source('results/experiments/m2_body_quality_v1/protocol.json')
    if sources['results/experiments/m2_body_quality_v1/protocol.json'] != body_protocol['prior_protocol_sha256']:
        raise ValueError('BODY reference protocol does not match inherited V2 bank')
    counts = Counter(s['role'] for s in body_protocol['samples'])
    construction.append({'view': 'M2螺杆 V2候选', 'normal_reference_images': sum(s['role']=='bank' for s in prior_body['samples']), 'fitting_images': counts['train'],
                         'calibration_images': counts['calibration'], 'evaluation_images': body['test_count'],
                         'note': '复用V1正常参考；另有200张资格检查和998张历史回归；整体门槛未通过。'})
    for suffix, metrics, population in [('_new', body['methods'][body['selected_mode']], '400张原Train预留图'),
                                        ('_regression', body['historical_regression']['current'], '998张历史官方Test回归图')]:
        data.append({'id': 'm2_body'+suffix, 'view': 'm2_body', 'name': 'M2螺杆 · V2研究候选',
                     'population': population, 'scope': '两分区必须同时查看；未通过全部门槛，未绑定线上质量模型',
                     'state': '未发布', 'metrics': checked_metrics(metrics), 'runtime_model_arrays_sha256': None})
    csem = source('results/experiments/csem_screw_pilot_v1/report.json')
    csem_protocol = source('results/experiments/csem_screw_pilot_v1/protocol.json')
    construction.append({'view': '一字槽头部候选', 'normal_reference_images': sum(s['role']=='fit' and s['label']=='normal' for s in csem_protocol['samples']),
                         'fitting_images': sum(s['role']=='fit' for s in csem_protocol['samples']),
                         'calibration_images': sum(s['role']=='calibration' for s in csem_protocol['samples']),
                         'evaluation_images': csem['paired_test_images'],
                         'note': '按实物分区，114张测试图仅38个实物；原始光照配对不充当新增实物。'})
    data.append({'id': 'csem_501', 'view': 'csem_slotted_head', 'name': '一字槽头部 · 研究候选',
                 'population': '38个实物，主光照501', 'scope': '另两种配对光照准确率相同；114张图仅38个实物；召回未过85%预设门槛',
                 'state': '未发布', 'metrics': checked_metrics(csem['by_illumination']['501'][csem['selected']]),
                 'runtime_model_arrays_sha256': None})
    manifest = source('data/external/csem_misd/pilot_manifest.json')
    datasets.append({'name': 'CSEM一字槽头部', 'images': len(manifest['images']), 'unit': '本轮配对光照图',
                     'physical_specimens': len(manifest['specimens']), 'quality_labels': True})
    appearance = source('results/experiments/screw_appearance_v3/report.json')
    catalogue = source('results/product_families/screw/appearance.json')
    datasets.append({'name': 'MVTec Screws外观资料', 'images': catalogue['summary']['samples'], 'unit': '裁图',
                     'source_frames': catalogue['summary']['source_frames'], 'types': catalogue['summary']['types'], 'quality_labels': False})
    route = source('results/experiments/screw_family_router_v1/report.json')
    matched = appearance['comparisons']['fresh_test']['v2']
    value = {'schema': 1, 'version': 'screw-delivery-evidence-v2', 'family_key': 'screw',
             'built_utc': datetime.now(timezone.utc).isoformat(), 'datasets': datasets, 'construction': construction, 'quality_cohorts': data,
             'routing': {'arrays_sha256': route['arrays_sha256'], **route['metrics'], 'scope': route['scope']},
             'appearance': {'arrays_sha256': appearance['arrays_sha256'], 'positive_references': appearance['reviewed_reference_count'],
                            'exclusion_references': appearance['exclusion_reference_count'],
                            'known': matched['known'], 'correct_accepted': round(matched['known']*matched['known_correct_acceptance']),
                            'unknown': matched['other'], 'unknown_accepted': matched['accepted_other'],
                            'scope': '原图分组留出304张；仅外观匹配，不是缺陷判定'},
             'source_file_sha256': sources,
             'limitations': ['不同分区和任务不能合并成一个总准确率。', '旧测试集用于回归，已参与开发的结果不称为新盲测。',
                             '多件自动定位、新型号全覆盖和真实产线效果尚未验证。', '监督判别使用来源缺陷标签，整套系统不称为完全零样本。'],
             'quality_permissions_changed': False}
    write(results/EVIDENCE, value)
    print({'built': EVIDENCE, 'quality_cohorts': len(data), 'datasets': len(datasets), 'model_changed': False}, flush=True)


if __name__ == '__main__':
    build()
