"""One screw-family inventory and traceable evidence, without quality permissions."""
import math
import hashlib
import json
import re
from pathlib import Path

from products.families import read
from review.workflow import file_hash

EVIDENCE = 'product_families/screw/delivery_evidence.json'
FLOW_SOURCES = {'src/web/family_inspection.py', 'src/perception/family_router.py',
                'src/review/disposition.py', 'src/interface/report.py'}


def checked_metrics(value):
    """Never display a percentage that disagrees with the reported counts."""
    matrix = value['confusion_matrix']
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError('Invalid confusion matrix')
    if any(type(n) is not int or n < 0 for row in matrix for n in row):
        raise ValueError('Confusion counts must be nonnegative integers')
    (tn, fp), (fn, tp) = matrix; n = tn+fp+fn+tp
    if not n or value.get('n', n) != n:
        raise ValueError('Evaluation count mismatch')
    result = {'n': n, 'normal': tn+fp, 'defect': fn+tp, 'accuracy': (tn+tp)/n,
              'recall': tp/(tp+fn) if tp+fn else None,
              'false_positive_rate': fp/(tn+fp) if tn+fp else None,
              'false_positives': fp, 'false_negatives': fn, 'confusion_matrix': matrix}
    for key in ('accuracy', 'recall', 'false_positive_rate'):
        if key in value and value[key] is not None and (result[key] is None or
                not math.isfinite(value[key]) or abs(result[key]-value[key]) > 1e-8):
            raise ValueError('Evaluation percentage disagrees with confusion counts: '+key)
    return result


def read_evidence(results):
    path = Path(results)/EVIDENCE
    if not path.is_file():
        return None
    data = read(path)
    if data.get('schema') != 1 or data.get('family_key') != 'screw':
        raise ValueError('Screw evidence schema mismatch')
    for cohort in data['quality_cohorts']:
        cohort['metrics'] = checked_metrics(cohort['metrics'])
    return data


def runtime_inventory(results, products, policy):
    results = Path(results); family = results/'product_families/screw'
    def artifact(pointer, name):
        relative = read(family/pointer)['directory']
        directory = (results/relative).resolve()
        if Path(relative).is_absolute() or '..' in Path(relative).parts or not directory.is_relative_to((results/'experiments').resolve()):
            raise ValueError('Invalid family artifact directory')
        meta = read(directory/name); arrays = (directory/meta['arrays']).resolve()
        if arrays.parent != directory or file_hash(arrays) != meta['arrays_sha256']:
            raise ValueError('Family reference integrity check failed')
        return meta
    routing = artifact('operational_router.json', 'router.json')
    appearance = artifact('appearance_active.json', 'appearance.json')
    configurations = {p['key']: p for p in products}; groups = []
    holds = policy.get('holds', {})
    for view in routing['views']:
        product = configurations.get(view.get('product_key')); held = False; model_hash = None
        if product:
            meta_path = results/'perception/optimized'/product['perception_artifact']
            meta = read(meta_path); arrays = (meta_path.parent/meta['arrays']).resolve()
            if arrays.parent != meta_path.parent.resolve():
                raise ValueError('Invalid quality artifact path')
            model_hash = file_hash(arrays)
            model_key = hashlib.sha256((json.dumps(product, sort_keys=True)+file_hash(meta_path)+model_hash).encode()).hexdigest()
            held = product['key']+':'+model_key in holds
        groups.append({'key': view['key'], 'name': view['name'], 'quality_model_bound': bool(product),
                       'product_key': product['key'] if product else None, 'automatic_hold': held,
                       'model_arrays_sha256': model_hash,
                       'threshold': product.get('threshold') if product else None,
                       't_high': product.get('t_high') if product else None,
                       'status': '已绑定检测 · 自动处置暂停' if held else '已绑定检测 · 继续执行作业门槛' if product else '质量候选未通过 · 转复核',
                       'reference_url': '/api/product-families/screw/inspection/references/'+view['key']})
    return {'family_key': 'screw', 'name': '螺丝族统一参考库',
            'organization': '一个族级入口与参考目录，共享CLIP编码器；内部按视图组织参考、判别头和阈值。当前复用两套质量配置，尚非所有外观共用一个质量模型。',
            'router_version': routing['version'], 'router_arrays_sha256': routing['arrays_sha256'],
            'appearance_version': appearance['version'], 'appearance_arrays_sha256': appearance['arrays_sha256'],
            'view_groups': groups, 'appearance_categories': [c['name'] for c in appearance['categories']],
            'appearance_reference_count': len(appearance['reference_keys']),
            'exclusion_reference_count': appearance.get('exclusion_reference_count', 0),
            'shared_encoder': 'CLIP ViT-B/16', 'automatic_permission_granted_by_inventory': False}


def showcase(results, products, policy, operational):
    inventory = runtime_inventory(results, products, policy)
    data = read_evidence(results)
    if data:
        groups = {g['key']: g for g in inventory['view_groups']}
        for cohort in data['quality_cohorts']:
            expected = cohort.get('runtime_model_arrays_sha256')
            current = groups.get(cohort['view'], {})
            cohort['matches_current_model'] = bool(expected and current.get('model_arrays_sha256') == expected and
                current.get('threshold') == cohort.get('runtime_threshold') and current.get('t_high') == cohort.get('runtime_t_high'))
        data['routing_matches_current'] = data['routing']['arrays_sha256'] == inventory['router_arrays_sha256']
        data['appearance_matches_current'] = data['appearance']['arrays_sha256'] == inventory['appearance_arrays_sha256']
    acceptance_path = Path(results)/'product_families/screw/delivery_acceptance.json'
    acceptance = read(acceptance_path) if acceptance_path.is_file() else None
    if acceptance and (acceptance.get('schema') != 1 or acceptance.get('passed') is not True or acceptance.get('record_scope') != 'simulation'):
        raise ValueError('Invalid integration acceptance record')
    if acceptance:
        if set(acceptance.get('runtime_source_sha256', {})) != FLOW_SOURCES:
            raise ValueError('Invalid workflow version evidence')
        if not re.fullmatch('[a-f0-9]{32}', acceptance['batch_id']) or any(
                not re.fullmatch('[0-9]{6}_[a-f0-9]{10}', c['session_id']) or c.get('pdf_valid') is not True
                for c in acceptance['checks']):
            raise ValueError('Invalid integration trace identity')
        root = Path(results).resolve().parent
        acceptance['matches_current_workflow'] = bool(acceptance.get('runtime_source_sha256')) and all(
            (root/name).is_file() and file_hash(root/name) == expected
            for name, expected in acceptance.get('runtime_source_sha256', {}).items())
    return {'ok': True, 'inventory': inventory, 'evidence_ready': data is not None, 'evidence': data, 'integration': acceptance,
            'operations': operational, 'workflow': [
                {'step': '01', 'title': '统一参考库', 'detail': '来源、外观与采集视图分组；正常参考与外观资料分别管理。'},
                {'step': '02', 'title': '自动匹配与检测', 'detail': '只选螺丝；匹配适用参考，输出评分、热力图，灰区执行AI分析。'},
                {'step': '03', 'title': '归档与复核', 'detail': '按有效作业资格处理；不确定、暂停版本与抽查进入复核，保存报告。'},
                {'step': '04', 'title': '评估与迭代', 'detail': '核验真值经批准进入候选；固定评估、人工应用或回滚，测试样本不回流。'}],
            'demo_note': '固定演示强制转核验，不计正式作业统计；离线检测成绩、路由成绩和人工一致性分别展示。'}
