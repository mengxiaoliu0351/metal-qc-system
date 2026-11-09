"""Appearance retrieval with source-labelled exclusion references; no quality truth."""
import numpy as np

from perception.appearance import AppearanceMatcher,ranked,unit


def boundary_ranked(features,references,labels,categories,exclusions):
    result=ranked(features,references,labels,categories)
    other=unit(np.atleast_2d(features))@unit(exclusions).T
    nearest=np.argmax(other,axis=1)
    result['exclusion_similarity']=other[np.arange(len(other)),nearest]
    result['exclusion_index']=nearest
    result['boundary_gap']=result['similarity']-result['exclusion_similarity']
    return result


class BoundaryAppearanceMatcher(AppearanceMatcher):
    def __init__(self,path):
        super().__init__(path)
        if self.meta.get('matcher_kind')!='source-exclusion-v1' or self.meta['selected_method']!='clip':raise ValueError('Invalid boundary matcher type')
        bank=self.params.get('exclusion_clip');labels=self.params.get('exclusion_labels')
        if bank is None or labels is None or bank.ndim!=2 or len(bank)!=len(labels) or len(bank)!=len(self.meta['exclusion_keys']):raise ValueError('Invalid exclusion reference arrays')
        if bank.shape[1]!=self.params['clip'].shape[1] or set(labels.tolist())&set(self.categories):raise ValueError('Exclusion labels overlap supported categories')
        fixed=self.meta.get('fixed_positive_gate')
        if fixed and (self.meta['gate']['similarity']<fixed['similarity'] or self.meta['gate']['margin']<fixed['margin']):
            raise ValueError('Conservative boundary cannot widen the baseline acceptance gate')

    def predict(self,features):
        values=boundary_ranked(features,self.params['clip'],self.params['labels'],self.categories,self.params['exclusion_clip'])
        value=super().predict(features);gap=float(values['boundary_gap'][0]);index=int(values['exclusion_index'][0])
        rejected=gap<self.meta['gate']['exclusion_margin']
        if rejected:value.update(category_id=None,category_name='未覆盖或匹配不确定',matched=False)
        return {**value,'exclusion_similarity':float(values['exclusion_similarity'][0]),'boundary_gap':gap,
            'exclusion_reference_key':self.meta['exclusion_keys'][index],
            'exclusion_category_id':int(self.params['exclusion_labels'][index]),
            'reason':'更接近或难以区分未覆盖类型' if rejected else ('已通过外观匹配门槛' if value['matched'] else '外观相似度或类别区分不足')}


def load_matcher(path):
    import json
    from pathlib import Path
    metadata=json.loads(Path(path).read_text(encoding='utf8'))
    kind=metadata.get('matcher_kind')
    if kind not in (None,'source-exclusion-v1'):raise ValueError('Unknown appearance matcher type')
    return BoundaryAppearanceMatcher(path) if kind else AppearanceMatcher(path)
