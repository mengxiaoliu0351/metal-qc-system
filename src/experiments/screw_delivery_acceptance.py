"""Read-only acceptance of the ordinary screw type and retained research evidence."""
import argparse
import csv
import io
from pathlib import Path

import requests

from products.families import write
from products.screw_delivery import checked_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); base = args.base.rstrip('/'); session = requests.Session(); session.trust_env = False
    def get(path):
        response = session.get(base+path, timeout=45); response.raise_for_status(); return response
    html = get('/').text
    assert 'screw-showcase-ui.js' not in html and 'screw-showcase.css' not in html
    app = get('/static/app.js').text
    assert 'ScrewShowcaseUI' not in app and '查看参考与匹配说明' in app
    assert '研发设置' in get('/static/product-ui.js').text
    targets = get('/api/products').json()['inspection_targets']
    assert [t['name'] for t in targets] == ['螺母', '螺丝', '金属网格']
    data = get('/api/product-families/screw/showcase').json()
    assert data['evidence_ready']; inventory = data['inventory']; evidence = data['evidence']
    assert len(inventory['view_groups']) == 3 and inventory['appearance_reference_count'] == 24 and inventory['exclusion_reference_count'] == 72
    assert next(v for v in inventory['view_groups'] if v['key'] == 'm2_head')['automatic_hold']
    assert not next(v for v in inventory['view_groups'] if v['key'] == 'm2_body')['quality_model_bound']
    assert len(evidence['quality_cohorts']) == 5 and len(evidence['construction']) == 4
    assert evidence['routing_matches_current'] and evidence['appearance_matches_current']
    for cohort in evidence['quality_cohorts']:
        checked_metrics(cohort['metrics'])
    assert all(c['matches_current_model'] for c in evidence['quality_cohorts'][:2])
    assert all(not c['matches_current_model'] for c in evidence['quality_cohorts'][2:])
    rows = list(csv.reader(io.StringIO(get('/api/product-families/screw/showcase/export').content.decode('utf-8-sig'))))
    assert len(rows) == 6 and any('未发布' in row for row in rows[1:])
    integration = data['integration']; assert integration['passed']
    assert len(integration['checks']) == 5 and integration['mixed_view_count'] == 3
    for check in integration['checks']:
        assert get('/api/report/'+check['session_id']).content.startswith(b'%PDF')
    write(args.output, {'passed': True, 'base': base, 'reference_views': 3, 'quality_cohorts': 5,
          'construction_rows': 4, 'confusion_counts_verified': True, 'current_model_identity_verified': True,
          'unreleased_candidates_kept_visible': True, 'csv_rows': 5, 'actual_demo_pdfs_verified': 5,
          'mixed_batch_items': 3, 'head_automatic_hold_preserved': True, 'body_quality_disabled': True,
          'formal_family_records': data['operations']['new_policy_records'],
          'user_interface': 'ordinary screw type; no standalone showcase mounted',
          'browser_visual_verification': 'not performed; source, API and frontend logic validated',
          'scope': 'metadata/static resources/export and real prior integration receipts; no new model accuracy measured'})
    print({'passed': True, 'cohorts': 5, 'traceable_demo_pdfs': 5, 'formal_family_records': data['operations']['new_policy_records']}, flush=True)


if __name__ == '__main__':
    main()
