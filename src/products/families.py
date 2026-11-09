"""Product-family identity, artifact isolation and reviewed appearance candidates.

The default screw family bridges the existing pilot. Registering a family or
importing category-only data never grants quality labels or model activation.
"""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import re
import threading
import uuid


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');temporary.replace(path)


def now():return datetime.now(timezone.utc).isoformat(timespec='seconds')


class FamilyRegistry:
    def __init__(self,results):
        self.results=Path(results);self.directory=self.results/'product_families'
        self.path=self.directory/'registry.json';self.lock=threading.RLock()

    @staticmethod
    def key(value):
        if not isinstance(value,str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,47}',value):raise ValueError('产品族标识无效')
        return value

    def state(self):
        if self.path.is_file():return read(self.path)
        return {'schema':1,'families':[{'key':'screw','name':'螺丝族','description':'多外观螺丝统一参考与研究试检',
            'encoder_product':'screw','model':None,'created':'legacy-pilot','automatic_archival':False}]}

    def get(self,key):
        key=self.key(key)
        result=next((f for f in self.state()['families'] if f['key']==key),None)
        if result is None:raise FileNotFoundError('产品族不存在')
        return result

    def folder(self,key):return self.directory/self.get(key)['key']

    def artifact(self,key):
        family=self.get(key);binding=family.get('model')
        if binding:relative=binding['directory']
        elif key=='screw':
            legacy=self.results/'perception/family_active.json'
            name=read(legacy).get('directory','') if legacy.is_file() else 'screw_family_v1'
            if not re.fullmatch(r'screw_family_v[0-9]+',name):raise ValueError('产品族版本配置无效')
            relative='experiments/'+name
        else:raise FileNotFoundError('该产品族尚未绑定研究模型')
        path=(self.results/relative).resolve()
        if Path(relative).is_absolute() or '..' in Path(relative).parts or not path.is_relative_to(self.results.resolve()):
            raise ValueError('产品族模型目录无效')
        return path

    def list(self):
        result=[]
        for f in self.state()['families']:
            item=dict(f)
            try:
                artifact=self.artifact(f['key'])
                item['model_ready']=(artifact/'family.json').is_file() and (artifact/'report.json').is_file()
            except FileNotFoundError:item['model_ready']=False
            appearance=self.directory/f['key']/'appearance.json'
            dataset=read(appearance) if appearance.is_file() else {}
            item['appearance_summary']=dataset.get('summary',{'types':0,'samples':0})
            checks=self.checks(f['key'])
            item['appearance_accepted']=sum(r['appearance_status']=='accepted' for r in checks.values())
            item['appearance_rejected']=sum(r['appearance_status']=='rejected' for r in checks.values())
            # This catalogue records appearance suitability, never quality truth.
            item['reviewed_normal']=0;item['eligible_normal_references']=0
            item['legacy_quality_marks']=sum(r.get('label') in ('normal','defect') for r in self.reviews(f['key']).values())
            item['status']='advisory' if item['model_ready'] else 'draft'
            result.append(item)
        return result

    def create(self,request):
        name=request.get('name','');description=request.get('description','')
        if not isinstance(name,str) or not 1<=len(name.strip())<=80:raise ValueError('请填写产品族名称，最多80字')
        if not isinstance(description,str) or len(description)>1000:raise ValueError('产品族说明过长')
        with self.lock:
            state=self.state()
            if len(state['families'])>=32:raise ValueError('最多登记32个产品族')
            if any(f['name']==name.strip() for f in state['families']):raise ValueError('已有同名产品族')
            family={'key':'family_'+uuid.uuid4().hex[:12],'name':name.strip(),'description':description.strip(),
                    'encoder_product':None,'model':None,'created':now(),'automatic_archival':False}
            state['families'].append(family);write(self.path,state)
        return family

    def bind(self,key,relative):
        self.get(key)
        path=(self.results/relative).resolve()
        if Path(relative).is_absolute() or '..' in Path(relative).parts or not path.is_relative_to(self.results.resolve()):raise ValueError('模型目录无效')
        meta=read(path/'family.json')
        if meta.get('family_key')!=key and not(key=='screw' and str(meta.get('version','')).startswith('screw-family-')):
            raise ValueError('模型所属产品族不匹配')
        arrays=(path/meta['arrays']).resolve()
        if arrays.parent!=path or hashlib.sha256(arrays.read_bytes()).hexdigest()!=meta['arrays_sha256']:raise ValueError('模型数组校验失败')
        if meta.get('automatic_archival') is not False:raise ValueError('当前登记只接受研究试检模型')
        with self.lock:
            state=self.state();family=next(f for f in state['families'] if f['key']==key)
            previous=family.get('model');family['model']={'directory':path.relative_to(self.results.resolve()).as_posix(),
                'version':meta['version'],'arrays_sha256':meta['arrays_sha256']}
            write(self.directory/key/'model_audit'/f'{uuid.uuid4().hex}.json',{'before':previous,'after':family['model'],'ts':now()})
            write(self.path,state)
        return family

    def appearance(self,key):
        path=self.folder(key)/'appearance.json'
        return read(path) if path.is_file() else {'summary':{'types':0,'samples':0},'types':[],'samples':[]}

    def reviews(self,key):
        """Preserve legacy annotations for audit; they do not establish quality."""
        path=self.folder(key)/'appearance_reviews.json'
        return read(path) if path.is_file() else {}

    def checks(self,key):
        path=self.folder(key)/'appearance_checks.json'
        return read(path) if path.is_file() else {}

    def reservations(self,key):
        path=self.folder(key)/'appearance_reservations.json'
        return read(path).get('samples',{}) if path.is_file() else {}

    def samples(self,key,category='',status='all',offset=0,limit=24):
        if offset<0 or not 1<=limit<=100 or status not in ('all','unreviewed','accepted','rejected','unverified','normal','defect'):raise ValueError('样本筛选参数无效')
        data=self.appearance(key);checks=self.checks(key);legacy=self.reviews(key);reserved=self.reservations(key);rows=[]
        for sample in data['samples']:
            if category and str(sample['category_id'])!=str(category):continue
            review=checks.get(sample['key'],{});state=review.get('appearance_status','unreviewed')
            if status in ('normal','defect'):continue
            if status in ('unreviewed','accepted','rejected') and state!=status:continue
            rows.append({**{k:v for k,v in sample.items() if k!='path'},'quality_label':'unverified',
                'appearance_status':state,'appearance_review':review,'legacy_quality_review_present':sample['key'] in legacy,
                'evaluation_reserved':sample['key'] in reserved,
                'eligible_normal_reference':False,
                'image_url':f"/api/product-families/{key}/appearance/{sample['key']}/image"})
        return {'ok':True,'summary':data['summary'],'types':data['types'],'dataset':data.get('dataset'),
            'count':len(rows),'offset':offset,'samples':rows[offset:offset+limit]}

    def image(self,key,sample_key):
        if not re.fullmatch(r'[a-f0-9]{64}',sample_key):raise ValueError('样本标识无效')
        sample=next((s for s in self.appearance(key)['samples'] if s['key']==sample_key),None)
        if not sample:raise FileNotFoundError('样本不存在')
        path=(self.results/sample['path']).resolve();scope=self.folder(key).resolve()
        if not path.is_relative_to(scope) or not path.is_file():raise ValueError('样本路径无效')
        return path

    def review(self,key,request):
        ids=request.get('sample_keys',[]);state=request.get('appearance_status');reviewer=request.get('reviewer','')
        if 'label' in request or 'quality_label' in request or request.get('use_for_bank') is True:
            raise ValueError('此入口仅审核类别、裁剪和清晰度，不能确认正常或缺陷，也不能批准正常参考；请刷新页面')
        if not isinstance(ids,list) or not 1<=len(ids)<=100 or any(not isinstance(s,str) for s in ids) or len(set(ids))!=len(ids):raise ValueError('请选择1至100个不重复样本')
        if state not in ('accepted','rejected','unreviewed'):raise ValueError('外观资料审核状态无效')
        if not isinstance(reviewer,str) or not reviewer.strip() or len(reviewer)>80:raise ValueError('请填写审核人')
        with self.lock:
            samples={s['key']:s for s in self.appearance(key)['samples']}
            for sample_key in ids:
                if sample_key not in samples:raise ValueError('核验样本不存在')
                sample=samples[sample_key]
                if state=='accepted' and sample.get('category_conflict'):raise ValueError('类别存在冲突，不能通过外观资料审核')
            reviews=self.checks(key);revision=uuid.uuid4().hex
            values={'appearance_status':state,'reviewer':reviewer.strip(),'note':str(request.get('note',''))[:2000],
                    'verification_method':'operator_appearance_check','purpose':'appearance_coverage_only','ts':now(),'revision':revision}
            for sample_key in ids:reviews[sample_key]=values.copy()
            write(self.folder(key)/'appearance_check_audit'/f'{revision}.json',{'sample_keys':ids,'review':values})
            write(self.folder(key)/'appearance_checks.json',reviews)
        return {'ok':True,'reviewed':len(ids),'model_changed':False,'quality_label_changed':False,'reference_approved':False,
            'note':'外观资料审核已保存；不作质量判定，不加入正常参考库，不修改模型或阈值'}
