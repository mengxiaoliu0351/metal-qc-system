"""One family choice -> guarded view routing -> existing production workflow or review.

Routing never activates a new quality model, clears a hold, or creates truth labels.
"""
import io
import json
from pathlib import Path
import re
import time

from fastapi import APIRouter,File,HTTPException,UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

from perception.appearance import full_object,unit
from perception.family_router import FamilyRouter
from products.families import read
from review.disposition import image_quality
from review.inspection_context import INSPECTION_CONTEXT
from review.workflow import file_hash

router=APIRouter();_routers={}
TARGETS={'mvtec_screw':'screw','m2_head':'product_a269405b7b9a'}


def inspection_targets(products,results):
    """User-facing objects, distinct from internal recipe identities."""
    definitions=[];folder=Path(results)/'product_families/screw';available=False
    try:
        pointer=read(folder/'operational_router.json');root=Path(results).resolve();directory=(root/pointer['directory']).resolve()
        if directory.is_relative_to((root/'experiments').resolve()):available=read(directory/'report.json')['operational_routing_ready'] is True
    except (OSError,ValueError,KeyError):pass
    for product in products:
        key=product['key']
        if key=='product_a269405b7b9a':continue
        if key=='screw':definitions.append({'key':'screw','name':'螺丝','mode':'family','family_key':'screw','batch_key':'family:screw','ready':available})
        else:definitions.append({'key':key,'name':{'metal_nut':'螺母','grid':'金属网格'}.get(key,product['name']),
            'mode':'product','product_key':key,'batch_key':key,'ready':True})
    return definitions


def batch_recipe(key,product_recipe):
    """Freeze the router plus every possible destination recipe for durable batches."""
    from web import server
    from web.family_api import get_appearance_model,registry
    model=load_router(key);folder=model.path.parent;report=read(folder/'report.json')
    if not report['operational_routing_ready']:raise ValueError('产品族自动匹配尚未通过验收')
    appearance=get_appearance_model(key);appearance_folder=appearance.path.parent
    paths=[folder/name for name in ('router.json','router.npz','protocol.json','report.json')]
    paths+=[appearance.path,appearance_folder/appearance.meta['arrays'],appearance_folder/'report.json',registry().folder(key)/'appearance_active.json']
    paths+=[Path(server.ROOT)/name for name in ('src/web/family_inspection.py','src/perception/family_router.py','src/perception/appearance.py','src/review/inspection_context.py')]
    return {'config':{'key':'family:'+key,'name':'螺丝','family_key':key,'mode':'family'},
        'files':{str(path.relative_to(server.ROOT)) if path.is_relative_to(Path(server.ROOT)) else str(path):file_hash(path) for path in paths},
        'destinations':{product:product_recipe(product) for product in TARGETS.values()}}


def load_router(key):
    from web import server
    from web.family_api import registry,operation
    operation(registry().get,key);pointer=registry().folder(key)/'operational_router.json'
    if not pointer.exists():raise FileNotFoundError('该产品族尚未完成统一接入')
    relative=read(pointer)['directory'];root=registry().results;folder=(root/relative).resolve()
    if Path(relative).is_absolute() or '..' in Path(relative).parts or not folder.is_relative_to((root/'experiments').resolve()):raise ValueError('产品族分流目录无效')
    files=[folder/name for name in ('router.json','router.npz','protocol.json','report.json')]
    runtime={field:Path(server.ROOT)/name for field,name in {'preprocessing_sha256':'src/perception/appearance.py','extractor_sha256':'src/perception/winclip_patch.py'}.items()}
    stamp=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in files+list(runtime.values()))
    if stamp not in _routers:
        model=FamilyRouter(files[0]);report=read(files[3]);meta=model.meta
        if meta['family_key']!=key or file_hash(files[2])!=meta['protocol_sha256'] or report['protocol_sha256']!=meta['protocol_sha256'] or report['arrays_sha256']!=meta['arrays_sha256']:
            raise ValueError('分流参考、协议和报告不配套')
        if any(file_hash(path)!=meta[field] for field,path in runtime.items()) or file_hash(Path(server.ROOT)/'weights/CLIP-ViT-B-16-open_clip_model.safetensors')!=meta['weights_sha256']:
            raise ValueError('分流参考与当前编码器或输入处理不配套')
        if key!='screw' or {v['key']:v.get('product_key') for v in meta['views']}!={**TARGETS,'m2_body':None}:raise ValueError('尚未验证该族的质量模型映射')
        _routers.clear();_routers[stamp]=model
    return _routers[stamp]


@router.get('/api/product-families/{key}/inspection')
def status(key:str):
    from web import server
    from web.family_api import registry,operation
    operation(registry().get,key)
    try:
        model=load_router(key);report=read(model.path.parent/'report.json');examples=read(model.path.parent/'demo_samples.json')['samples']
    except FileNotFoundError:return {'ok':True,'ready':False,'message':'该产品族尚未完成统一接入，不会套用其他族的模型'}
    except (OSError,ValueError,KeyError) as exc:raise HTTPException(503,'产品族统一配置校验失败') from exc
    products={p['key']:p for p in server.load_products()}
    return {'ok':True,'ready':report['operational_routing_ready'],'version':model.meta['version'],'report':report,
        'views':[{**v,'reference_url':f"/api/product-families/{key}/inspection/references/{v['key']}",'quality_model_bound':v.get('product_key') in products,
            'status':'接入已有检测与作业门槛' if v.get('product_key') in products else '质量候选未通过门槛，仅转复核'} for v in model.meta['views']],
        'examples':[{**{k:v for k,v in s.items() if k!='file'},'image_url':f"/api/product-families/{key}/inspection/examples/{s['id']}"} for s in examples],
        'scope':'只自动匹配已覆盖的来源／视图，不识别具体型号；原质量门槛、暂停和抽查规则继续执行'}


def example_path(key,sample_id):
    if not re.fullmatch('[a-f0-9]{64}',sample_id):raise HTTPException(400,'演示样例编号无效')
    try:
        folder=load_router(key).path.parent;sample=next((s for s in read(folder/'demo_samples.json')['samples'] if s['id']==sample_id),None)
        if not sample:raise HTTPException(404,'样例不存在')
        path=(folder/sample['file']).resolve()
        if not path.is_relative_to((folder/'demo_images').resolve()) or file_hash(path)!=sample['file_sha256']:raise ValueError('演示图校验失败')
        return path
    except (OSError,ValueError,KeyError) as exc:raise HTTPException(503,'演示样例尚未就绪') from exc


@router.get('/api/product-families/{key}/inspection/examples/{sample_id}')
def example(key:str,sample_id:str):return FileResponse(example_path(key,sample_id))


@router.get('/api/product-families/{key}/inspection/references/{view}')
def reference_image(key:str,view:str):
    try:
        folder=load_router(key).path.parent;entry=next((s for s in read(folder/'references.json')['items'] if s['view_key']==view),None)
        if not entry:raise HTTPException(404,'参考视图不存在')
        path=(folder/entry['file']).resolve()
        if not path.is_relative_to((folder/'reference_images').resolve()) or file_hash(path)!=entry['file_sha256']:raise ValueError('参考图校验失败')
        return FileResponse(path)
    except (OSError,ValueError,KeyError) as exc:raise HTTPException(503,'参考图尚未就绪') from exc


def finish_pending(image,filename,started,reason,context,quality=None):
    from web import server
    context={**context,'product_key':'family_'+context['family_key']+'_unassigned','product_name':context.get('family_name',context['family_key'])+' · 待适用性／质量核验',
        'family_action':'review_only','quality_model_bound':False,
        'perception_version':'未执行质量模型','iteration_eligible':False}
    token=INSPECTION_CONTEXT.set(context)
    try:
        response=server._finalize_detection(image,filename,None,{},started,{'ok':True,'score':None,'threshold':None,'t_high':None,
            'route':'family_review','route_label':'产品族例外复核','vlm_called':False,'heatmap':None,'perception_prediction':None,
            'cognition':{'ok':False,'conclusion':'待人工核验','defect_desc':reason,'suggestion':'先核对适用视图和真实质量；本次没有判定为正常或缺陷'}})
        data=json.loads(response.body);data.update(family_key=context['family_key'],family_routing=context.get('family_routing'),
            family_action='review_only',quality_model_bound=False,iteration_eligible=False)
        return data
    finally:INSPECTION_CONTEXT.reset(token)


def inspect(content,filename,key,simulation=False):
    from web import server
    from web.family_api import registry,operation,get_appearance_model
    started=time.perf_counter();family=operation(registry().get,key)
    try:
        with Image.open(io.BytesIO(content)) as original:
            if original.format not in ('PNG','JPEG','BMP','WEBP') or original.width*original.height>20_000_000:raise ValueError('请上传不超过2000万像素的常用图片')
            image=original.convert('RGB')
    except (OSError,ValueError,Image.DecompressionBombError) as exc:raise HTTPException(400,str(exc)) from exc
    context={'family_key':key,'family_name':family['name'],'family_entry_version':'unified-inspection-v1','_started':started}
    if simulation:context['record_scope']='simulation'
    quality=image_quality(image)
    if not quality['ok']:
        return finish_pending(image,filename,started,'图像质量待核验：'+'；'.join(quality['reasons']),{**context,'family_routing':{'matched':False,'reason':'输入图像质量未通过'}},quality)
    with server._inference_lock:
        try:model=load_router(key);report=read(model.path.parent/'report.json')
        except (OSError,ValueError,KeyError) as exc:raise HTTPException(503,'统一匹配参考尚未就绪') from exc
        if not report['operational_routing_ready']:raise HTTPException(409,'统一匹配尚未通过分流验收，暂不调用质量模型')
        detector,_=server.get_engine('screw');_,vector=detector.extract_features(full_object(image));routing=model.predict(vector)
        # Appearance approval does not authorize quality classification.
        try:appearance=get_appearance_model(key).predict(vector)
        except (OSError,ValueError,KeyError):appearance=None
        if appearance and appearance['matched']:
            routing={**routing,'matched':False,'view_key':None,'view_name':appearance['category_name'],
                'reason':'已匹配外观资料，但该外观没有正常／缺陷验证，不能套用已有质量模型','appearance':appearance}
        if routing.get('view_key'):routing['reference_image_url']=f"/api/product-families/{key}/inspection/references/{routing['view_key']}"
        context.update(family_routing=routing,family_routing_seconds=round(time.perf_counter()-started,3))
        if not routing['matched']:return finish_pending(image,filename,started,routing['reason'],context)
        product=TARGETS.get(routing['view_key'])
        if not product:return finish_pending(image,filename,started,'已识别M2螺杆视图，但质量候选未通过独立门槛，本次不作正常／缺陷判定',context)
        if product not in {p['key'] for p in server.load_products()}:raise HTTPException(503,'已匹配视图尚无成套检测配置')
        context.update(family_action='product_workflow',quality_model_bound=True)
        token=INSPECTION_CONTEXT.set(context)
        try:response=server._locked_detection(content,filename,product)
        finally:INSPECTION_CONTEXT.reset(token)
    data=json.loads(response.body)
    if response.status_code!=200:raise HTTPException(response.status_code,data.get('error','对应检测流程执行失败'))
    return {**data,'family_key':key,'family_routing':routing,'product_key':product,'family_action':'product_workflow','quality_model_bound':True,
        'scope':'自动匹配后复用已有产品检测、灰区分析、作业门槛、抽查和历史；匹配成功不代表质量合格'}


@router.post('/api/product-families/{key}/inspection/detect')
async def detect(key:str,file:UploadFile=File(...)):
    content=await file.read(10*1024*1024+1)
    if not content or len(content)>10*1024*1024:raise HTTPException(400,'图片须小于10MB')
    return await run_in_threadpool(inspect,content,file.filename,key)


@router.post('/api/product-families/{key}/inspection/examples/{sample_id}/detect')
async def example_detect(key:str,sample_id:str):
    path=await run_in_threadpool(example_path,key,sample_id)
    return await run_in_threadpool(inspect,path.read_bytes(),'产品族固定演示_'+sample_id[:8]+'.png',key,True)


@router.get('/api/product-families/{key}/inspection/metrics')
def metrics(key:str):
    from web import server
    from web.family_api import registry,operation
    from review.metrics import operational_summary
    operation(registry().get,key);records=[]
    for _,record in server.get_review_workflow().records():
        if record.get('family_key')==key and record.get('record_scope','production')=='production':records.append(record)
        if len(records)>=500:break
    summary=operational_summary(records)
    summary['scope_note']='本族最近500条作业记录；核验一致性受例外与抽查选择影响，不能当作独立测试准确率'
    return {'ok':True,'family_key':key,'summary':summary,'review_only_unmatched':sum(r.get('iteration_eligible') is False for r in records)}
