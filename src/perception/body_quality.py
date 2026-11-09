"""Research-only M2 body classifier, guarded by a separately calibrated view check."""
import hashlib
import json
from pathlib import Path

import numpy as np

from perception.appearance import unit
from perception.feature_backends import binary_probability,cosine_anomalies


def body_scores(vectors,raw,params,mode):
    if mode not in ('normal_memory','global_binary','fast_fusion'):raise ValueError('Unknown body quality mode')
    if mode=='normal_memory':return np.asarray(raw,dtype=float)
    global_score=binary_probability(vectors,params,'global')
    if mode=='global_binary':return global_score
    if mode=='fast_fusion':return binary_probability(np.c_[raw,global_score],params,'fusion')
    raise ValueError('Unknown body quality mode')


def scope_values(vectors,params):
    x=unit(np.atleast_2d(vectors));body=x@unit(params['body_centroid']);head=x@unit(params['head_centroid'])
    return body,head,body-head


class BodyQualityModel:
    def __init__(self,path):
        self.path=Path(path);self.meta=json.loads(self.path.read_text(encoding='utf8'));array=(self.path.parent/self.meta['arrays']).resolve()
        if array.parent!=self.path.parent.resolve() or hashlib.sha256(array.read_bytes()).hexdigest()!=self.meta['arrays_sha256']:raise ValueError('BODY model integrity failure')
        if self.meta.get('automatic_archival') is not False or self.meta.get('quality_label_basis')!='publisher_OK_NG':raise ValueError('BODY trial must preserve research-only quality provenance')
        with np.load(array,allow_pickle=False) as z:self.params={k:z[k].copy() for k in z.files}

    def predict(self,vector,raw):
        body,head,gap=scope_values(vector,self.params);gate=self.meta['view_gate']
        in_scope=bool(body[0]>=gate['min_body_similarity'] and gap[0]>=gate['margin'])
        score=float(body_scores(np.asarray(vector).reshape(1,-1),[raw],self.params,self.meta['selected_mode'])[0]);low,high=self.meta['thresholds'][self.meta['selected_mode']]
        return {'version':self.meta['version'],'scope_ok':in_scope,'body_similarity':float(body[0]),'head_similarity':float(head[0]),
            'recognized_view':'m2_body' if in_scope else ('m2_head_reference' if head[0]>body[0] else 'uncertain'),
            'score':score if in_scope else None,'normal_defect_hint':('defect' if score>=low else 'normal') if in_scope else None,
            'risk':('low' if score<low else 'high' if score>high else 'gray') if in_scope else 'unsupported',
            'threshold':low,'t_high':high,'requires_review':True,'automatic_archival':False,
            'quality_assessment':'research_only' if in_scope else 'not_available','method':self.meta['selected_mode']}

    def raw_scores(self,patches):return cosine_anomalies(patches,self.params['bank'])
