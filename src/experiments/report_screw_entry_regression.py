"""Export historical whole-entry regression without inventing an overall accuracy."""
import csv
from collections import Counter

from experiments.screw_entry_regression import OUTPUT, summarize
from products.families import read

NAMES = {'mvtec_screw':'螺丝近景历史留出','m2_head':'M2头部官方Test',
         'm2_body':'M2螺杆主回归','m2_body_preview':'M2螺杆已见预览图',
         'csem_uncovered':'一字槽头部未覆盖视图','appearance_no_quality_truth':'无质量标签的外观留出'}


def percentage(value):
    return '—' if value is None else f'{100*value:.2f}%'


def main():
    report = summarize(OUTPUT); protocol = read(OUTPUT/'protocol.json'); environment = read(OUTPUT/'environment.json')
    if not report['complete']:
        raise ValueError('Do not export partial processing as complete regression')
    rows = [read(path) for path in sorted((OUTPUT/'cases').glob('*.json'))]
    columns = ['来源/分区','图片数','有效质量判断','未获适用质量判断','已判断准确率','全体有标签图正确判定占比','误报','漏检','未判定缺陷','误套质量模型','转人工','VLM触发','VLM成功','中位后端耗时秒','P95后端耗时秒',
               '自动建议','其中抽查复核','免本次复核自动处置','自动建议错误','VLM触发率','已判断缺陷召回','已判断正常误报率']
    lines = ['# 螺丝统一入口完整历史回归', '',
             f'固定清单{report["expected"]}张，实际处理{report["processed"]}张。直接调用网站相同后端入口；所有输出保存在本机隔离目录，没有修改公网模型、策略或作业历史。', '',
             '这是已有公开数据的历史回归，不是新厂家、真实产线或新增实物盲测。没有重新训练、改阈值或提交人工判定。', '',
             '## 完整入口表现', '',
             '| 来源／分区 | 图片数 | 有效质量判断 | 未获适用质量判断 | 已判断准确率 | 全体有标签图正确判定占比 | 误报／漏检 | 未判定缺陷 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    csv_rows = []
    for key,n in protocol['cohort_counts'].items():
        m = report['cohorts'][key]
        if m['processed'] != n: raise ValueError('Cohort count changed')
        mistakes = f'{m["false_positives"]}／{m["false_negatives"]}' if m['valid_classified'] else '—'
        pending_defects = m['defects_abstained'] if m['quality_labelled'] else '—'
        part = [r for r in rows if r['sample']['cohort'] == key and r['status'] == 'completed']
        automatic = [r for r in part if r['result'].get('disposition',{}).get('auto_result') in ('normal','defect')]
        sampled = sum(bool(r['result']['disposition'].get('requires_review')) for r in automatic)
        (tn,fp),(fn,tp) = m['confusion_matrix_classified_only']
        lines.append(f'| {NAMES[key]} | {n} | {m["valid_classified"]} | {m["quality_abstained"]} | {percentage(m["accuracy_classified_only"])} | {percentage(m["correct_quality_decisions_over_labelled_all"])} | {mistakes} | {pending_defects} |')
        csv_rows.append([NAMES[key],n,m['valid_classified'],m['quality_abstained'],m['accuracy_classified_only'],
                         m['correct_quality_decisions_over_labelled_all'],m['false_positives'],m['false_negatives'],m['defects_abstained'],
                         m['misapplied_quality_models'],m['sent_to_review'],m['vlm_triggered'],m['vlm_successful'],
                         m['median_backend_wall_seconds'],m['p95_backend_wall_seconds'],len(automatic),sampled,len(automatic)-sampled,
                         m['automatic_wrong'] if automatic and m['quality_labelled'] else None,m['vlm_triggered']/n,
                         tp/(tp+fn) if tp+fn else None,fp/(tn+fp) if tn+fp else None])
    lines += ['', '“有效质量判断”要求参考视图正确、存在质量模型、来源有真值且返回正常／缺陷。误套模型即使碰巧猜中二分类标签，也不计为有效正确判断。', '',
              '“未获适用质量判断”单独列出，不作为正确正常，也不直接冒充算法漏检；其中真实缺陷另列为“未判定缺陷”。无质量真值的外观图不计算缺陷准确率。', '',
              '## 缺陷召回与正常误报', '',
              '| 来源／分区 | 来源正常／缺陷 | 已判断图缺陷召回 | 已判断图正常误报率 | 全部来源缺陷中被正确检出的占比 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for key,m in report['cohorts'].items():
        if not m['quality_labelled']: continue
        truth = Counter(s['truth'] for s in protocol['samples'] if s['cohort'] == key)
        (tn,fp),(fn,tp) = m['confusion_matrix_classified_only']
        recall = tp/(tp+fn) if tp+fn else None
        fpr = fp/(tn+fp) if tn+fp else None
        lines.append(f'| {NAMES[key]} | {truth["normal"]}／{truth["defect"]} | {percentage(recall)} | {percentage(fpr)} | {percentage(tp/truth["defect"] if truth["defect"] else None)} |')
    lines += ['', '最后一列把未获判断的缺陷也放进分母，反映当前入口的缺陷检出覆盖；它不是含人工结论的最终召回。没有运行质量模型的分区，0%仅表示系统未自动检出，不表示算法全漏检。', '',
              '## 作业分流、VLM和耗时', '',
              '| 来源／分区 | 转人工 | 自动建议 | 其中抽查复核 | 无需本次复核的自动处置 | 自动建议错误 | VLM触发／成功 | 后端中位耗时 | 后端P95 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key,m in report['cohorts'].items():
        part = [r for r in rows if r['sample']['cohort'] == key and r['status'] == 'completed']
        automatic = [r for r in part if r['result'].get('disposition',{}).get('auto_result') in ('normal','defect')]
        sampled = sum(bool(r['result']['disposition'].get('requires_review')) for r in automatic)
        auto_wrong = m['automatic_wrong'] if automatic and m['quality_labelled'] else '—'
        lines.append(f'| {NAMES[key]} | {m["sent_to_review"]} | {m["automatic_proposals"]} | {sampled} | {len(automatic)-sampled} | {auto_wrong} | {m["vlm_triggered"]}／{m["vlm_successful"]} | {m["median_backend_wall_seconds"]:.3f}s | {m["p95_backend_wall_seconds"]:.3f}s |')
    lines += ['', '“自动建议”和“转人工”存在抽查交集，不能直接相加。无需本次复核的自动处置指系统已归档／隔离且本轮没有抽中复核，不等于最终质量签发。未运行人工复核，因此没有最终人机系统准确率。原M2头部暂停和灰区资格门槛均保持不变。', '',
              f'运行环境：{environment["device"]}，PyTorch {environment["torch"]}，{environment["torch_threads"]}个计算线程；VLM为{environment["vlm_model"]}。', '',
              '耗时来自本机串行后端调用，包含模型首次加载、记录与图片落盘，以及真实VLM网络往返；没有经过浏览器或公网反向代理，不能当作阿里云服务器或产线节拍。', '',
              '## 路由和转人工原因', '']
    for key,m in report['cohorts'].items():
        review_reasons = Counter()
        for row in rows:
            if row['sample']['cohort'] != key or row['status'] != 'completed': continue
            action = row['result'].get('disposition',{})
            if action.get('requires_review'):
                review_reasons[action.get('reason_code') or ('periodic_sample' if action.get('sampled') else action.get('status','unspecified'))] += 1
        lines += [f'### {NAMES[key]}', '', f'- 路由分布：{m["routing"]}',
                  f'- 转人工原因（仅计实际转人工记录）：{dict(review_reasons)}。',
                  f'- 误套质量模型：{m["misapplied_quality_models"]}；处理错误：{m["errors"]}。', '']
    lines += ['## 怎样解读与下一步', '',
              '1. 统一入口解决用户不必挑选内部检测配置的问题；完整产品族的质量覆盖仍未完成。',
              '2. 当前近景和头部存在可运行的质量模型；螺杆和未覆盖外观按保护规则转复核，不能把这些拒判计作检测正确。',
              '   外观组473张中415张因分辨率不足，在图像质量检查阶段就转复核；其余58张也未获适用质量模型。因此这组只能说明没有强行输出质量结论，不能证明分流器在473张上都正确识别了未知外观。',
              '3. 优先补齐螺杆已有候选未通过的误报／召回门槛，再评估是否绑定到这个现有入口；继续使用已有开发分区，不新增网站入口。',
              '4. 头部保留自动处置暂停，候选必须先通过冻结对照才能恢复；不为提高自动化比例而放宽暂停规则。',
              '5. 不再为了展示堆数据、开新页面或反复调整Test阈值。按目前结果明确展示可用范围和失败范围。', '']
    lines += ['## 数据与证据边界', '',
              '- 近景只评价既有固定40张留出，原训练和校准图片未混入成绩。',
              '- M2头部536张、螺杆1000张均来自原官方Test；2张早期预览图单列，另外998张是主回归。',
              '- 一字槽114张是38个实物的三种配对光照，不是114个独立实物。',
              '- 473张外观裁图没有质量真值；它们用于检查是否强行套用质量模型。',
              '- 来源没有实物编号的部分只能按图像身份排重，不能保证跨实物独立。',
              '- 外观图的分组字段来自原始帧，不是实物编号；只有CSEM保留作者提供的实物编号。',
              '- 近景自动处置门槛以前已经参考过这批历史图，其自动建议错误数只能作回归观察，不能当作独立误判率保证。',
              '- 本轮没有调整模型或阈值；结果用于定位后续优化目标，不能追溯改写旧实验。', '',
              f'协议SHA256：`{report["protocol_sha256"]}`。逐图结果、隔离历史与汇总位于`results/experiments/screw_entry_regression_v1/`。']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    with (OUTPUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(columns);writer.writerows(csv_rows)
    with (OUTPUT/'cases.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['分区','来源图片','来源真值','预期视图','实际视图','有效质量判断','误套质量模型','本地判断','分数','风险分流','自动建议','转人工','原因','VLM触发','VLM成功','后端秒','记录编号','处理状态'])
        for row in rows:
            s=row['sample'];r=row['result'];action=r.get('disposition',{})
            writer.writerow([NAMES[s['cohort']],s['path'],s['truth'],s['expected_view'],r.get('family_routing',{}).get('view_key'),
                row['valid_quality_prediction'],row.get('misapplied_quality_model'),r.get('perception_prediction'),r.get('score'),r.get('route'),action.get('auto_result'),
                action.get('requires_review'),action.get('reason_code') or ('periodic_sample' if action.get('sampled') else action.get('status')),r.get('vlm_called'),r.get('cognition',{}).get('ok'),
                row['wall_seconds'],r.get('session_id'),row['status']])
    print({'report':str(OUTPUT/'report.md'),'csv':str(OUTPUT/'summary.csv')},flush=True)


if __name__=='__main__':main()
