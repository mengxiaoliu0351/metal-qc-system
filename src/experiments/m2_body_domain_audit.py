"""Exploratory source-partition separability, NOT a quality model or causal proof."""
import argparse
from pathlib import Path

import numpy as np

from experiments import m2_body_detail as detail
from experiments.m2_body_quality import load_features
from experiments.prepare_m2_body import sha
from products.families import read, write


def audit(output):
    if output.exists():
        raise ValueError('Domain audit is frozen; do not overwrite')
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from threadpoolctl import threadpool_limits
    p = read(detail.OUTPUT/'protocol.json')
    rows = [s for s in p['samples'] if s['label'] == 'normal' and s['role'] in ('calibration', 'regression')]
    diagnosis = {s['pixel_hash']: s for s in read(detail.OUTPUT/'crop_diagnosis.json')['rows']}
    clip, _ = load_features(detail.PRIOR, rows)
    geometry = []
    for s in rows:
        x1, y1, x2, y2 = diagnosis[s['pixel_hash']]['old_box']
        geometry.append([(x2-x1)/s['width'], (y2-y1)/s['height'],
                         (x1+x2)/(2*s['width']), (y1+y2)/(2*s['height'])])
    y = np.array([s['role'] == 'regression' for s in rows], int)
    reports = {}
    for name, x in [('crop_geometry', np.asarray(geometry)), ('frozen_clip', clip)]:
        pipeline = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(C=.01,
                          class_weight='balanced', max_iter=2000, random_state=20261107))])
        with threadpool_limits(limits=2):
            score = cross_val_predict(pipeline, x, y, method='predict_proba',
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=20261107))[:, 1]
        reports[name] = {'source_partition_oof_auroc': float(roc_auc_score(y, score)),
                         'source_partition_oof_balanced_accuracy': float(balanced_accuracy_score(y, score >= .5))}
    report = {'normal_calibration_n': int(sum(y == 0)), 'normal_historical_test_n': int(sum(y == 1)),
              'diagnostic': reports, 'prediction_target': 'calibration versus historical Test source partition, NOT normal/defect',
              'method': 'fixed C=.01, five image-stratified out-of-fold predictions; scaler fitted in each fold',
              'causal_claim': False, 'quality_accuracy_claim': False, 'eligible_for_production': False,
              'limitation': 'exploratory, previously seen data; physical identity unknown; separability does not identify lighting/background/shape as the cause',
              'protocol_sha256': sha(detail.OUTPUT/'protocol.json'), 'implementation_sha256': sha(Path(__file__))}
    write(output, report); print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=detail.OUTPUT/'domain_audit.json')
    args = parser.parse_args(); audit(args.output)


if __name__ == '__main__':
    main()
