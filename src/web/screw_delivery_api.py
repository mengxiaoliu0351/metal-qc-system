"""Read-only screw-family delivery evidence, distinct from changing live models."""
import csv
import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from products.screw_delivery import showcase

router = APIRouter()


def snapshot():
    from web import server
    from web.family_inspection import metrics
    try:
        summary = metrics('screw')['summary']
        return showcase(server.RESULTS_DIR, server.load_products(),
                        server.get_review_workflow().policy().settings(), summary)
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(503, '螺丝族库或评估依据校验失败，暂不展示成绩') from exc


@router.get('/api/product-families/screw/showcase')
def overview():
    return snapshot()


@router.get('/api/product-families/screw/showcase/export')
def export():
    data = snapshot()
    if not data['evidence_ready']:
        raise HTTPException(409, '螺丝族评估依据尚未生成')
    stream = io.StringIO(newline=''); writer = csv.writer(stream)
    writer.writerow(['视图/版本', '评价样本', '样本数', '正常数', '缺陷数', '准确率', '缺陷召回', '正常误报率', '误报', '漏检', '当前模型匹配', '状态', '适用口径'])
    for row in data['evidence']['quality_cohorts']:
        m = row['metrics']
        writer.writerow([row['name'], row['population'], m['n'], m['normal'], m['defect'], m['accuracy'],
                         m['recall'], m['false_positive_rate'], m['false_positives'], m['false_negatives'],
                         '是' if row['matches_current_model'] else '研究或历史版本', row['state'], row['scope']])
    return Response('\ufeff'+stream.getvalue(), media_type='text/csv; charset=utf-8',
                    headers={'Content-Disposition': 'attachment; filename="screw-family-evaluation.csv"'})
