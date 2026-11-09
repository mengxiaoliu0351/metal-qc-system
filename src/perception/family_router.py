"""Source/view suitability routing only; this class never judges part quality."""
import hashlib
from pathlib import Path

import numpy as np

from perception.appearance import unit
from products.families import read


class FamilyRouter:
    def __init__(self,path):
        self.path=Path(path);self.meta=read(self.path);target=(self.path.parent/self.meta['arrays']).resolve()
        if target.parent!=self.path.parent.resolve() or hashlib.sha256(target.read_bytes()).hexdigest()!=self.meta['arrays_sha256']:
            raise ValueError('Family routing reference integrity failure')
        if self.meta.get('quality_decision_permission') is not False:raise ValueError('Routing must not grant quality permission')
        with np.load(target,allow_pickle=False) as data:self.params={k:data[k].copy() for k in data.files}
        if len(self.meta['views'])!=len(self.params['centroids']):raise ValueError('Routing view/reference mismatch')

    def predict(self,vector):
        x=unit(np.asarray(vector).reshape(-1));scores=x@unit(self.params['centroids']).T;order=np.argsort(-scores);i=int(order[0])
        similarity=float(scores[i]);margin=float(scores[i]-scores[order[1]]);excluded=float((x@unit(self.params['exclusions']).T).max())
        thresholds=self.meta['gates'][i];matched=similarity>=thresholds['similarity'] and margin>=thresholds['margin'] and similarity>=excluded
        return {'matched':bool(matched),'view_key':self.meta['views'][i]['key'] if matched else None,
            'view_name':self.meta['views'][i]['name'] if matched else '未覆盖或匹配不确定',
            'nearest_view':self.meta['views'][i]['key'],'similarity':similarity,'margin':margin,'exclusion_similarity':excluded,
            'reason':'已匹配来源／视图参考' if matched else '参考不足、视图接近或更接近未覆盖外观，未选择质量模型',
            'router_version':self.meta['version'],'quality_decision_permission':False}
