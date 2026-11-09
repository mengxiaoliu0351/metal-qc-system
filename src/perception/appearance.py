"""Category-only product-family retrieval. It never predicts normal or defect."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageOps


def unit(x):
    x=np.asarray(x,dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('Non-finite appearance features')
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def full_object(image):
    """Keep the entire narrow crop instead of CLIP's centre square cropping."""
    return ImageOps.pad(image.convert('RGB'),(224,224),method=Image.Resampling.LANCZOS,color=(128,128,128))


def color_shape(image):
    image=image.convert('RGB');x=np.asarray(image.resize((16,32),Image.Resampling.BILINEAR),dtype=np.float32)/127.5-1
    gray=x.mean(axis=2);dx=np.diff(gray,axis=1);dy=np.diff(gray,axis=0)
    aspect=np.repeat(np.log(max(image.height,1)/max(image.width,1)),32).astype(np.float32)
    return unit(np.r_[x.ravel(),2*dx.ravel(),2*dy.ravel(),aspect])


def ranked(features,references,labels,categories):
    x=unit(np.atleast_2d(features));bank=unit(references)
    sims=x@bank.T
    scores=np.stack([sims[:,np.asarray(labels)==c].max(axis=1) for c in categories],axis=1)
    order=np.argsort(-scores,axis=1,kind='stable');first=order[:,0]
    return {'category':np.asarray(categories)[first],'similarity':scores[np.arange(len(x)),first],
        'margin':scores[np.arange(len(x)),first]-scores[np.arange(len(x)),order[:,1]],'scores':scores}


class AppearanceMatcher:
    def __init__(self,path):
        self.path=Path(path);self.meta=json.loads(self.path.read_text(encoding='utf8'))
        target=(self.path.parent/self.meta['arrays']).resolve()
        if target.parent!=self.path.parent.resolve() or hashlib.sha256(target.read_bytes()).hexdigest()!=self.meta['arrays_sha256']:
            raise ValueError('Appearance reference integrity failure')
        if self.meta.get('quality_labels_used') is not False or self.meta.get('automatic_archival') is not False:
            raise ValueError('Appearance matching cannot authorize quality decisions')
        with np.load(target,allow_pickle=False) as data:self.params={k:data[k].copy() for k in data.files}
        self.categories=[c['category_id'] for c in self.meta['categories']]

    def predict(self,features):
        values=ranked(features,self.params[self.meta['selected_method']],self.params['labels'],self.categories)
        category=int(values['category'][0]);similarity=float(values['similarity'][0]);margin=float(values['margin'][0])
        gate=self.meta['gate'];accepted=similarity>=gate['similarity'] and margin>=gate['margin']
        return {'category_id':category if accepted else None,'suggested_category_id':category,
            'category_name':next(c['name'] for c in self.meta['categories'] if c['category_id']==category) if accepted else '未覆盖或匹配不确定',
            'similarity':similarity,'margin':margin,'matched':bool(accepted),'quality_assessment':'not_available',
            'automatic_archival':False,'version':self.meta['version'],'method':self.meta['selected_method']}
