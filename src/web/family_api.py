"""Product-family catalogue and advisory inference, separate from production archival."""
import io
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from perception.family import FamilyModel
from review.disposition import image_quality
from review.workflow import file_hash, read
from products.families import FamilyRegistry
from perception.appearance import AppearanceMatcher,full_object,color_shape,unit
from perception.appearance_boundary import load_matcher

router = APIRouter()
_model_cache = {}
_registries={}
_appearance_cache={}


def registry():
    from web import server
    key=str(Path(server.RESULTS_DIR).resolve())
    if key not in _registries:_registries[key]=FamilyRegistry(key)
    return _registries[key]


def operation(function,*args,**kwargs):
    try:return function(*args,**kwargs)
    except FileNotFoundError as exc:raise HTTPException(404,str(exc)) from exc
    except ValueError as exc:raise HTTPException(400,str(exc)) from exc


def folder(family='screw'):
    try:return registry().artifact(family)
    except ValueError as exc:raise HTTPException(503,str(exc)) from exc


@router.get('/api/product-families')
def families():return {'ok':True,'families':operation(registry().list)}


@router.post('/api/product-families')
def family_create(request:dict):return {'ok':True,'family':operation(registry().create,request)}


@router.get('/api/product-families/{key}/appearance')
def appearance(key:str,category:str='',status:str='all',offset:int=0,limit:int=24):
    return operation(registry().samples,key,category,status,offset,limit)


@router.get('/api/product-families/{key}/appearance/{sample_key}/image')
def appearance_image(key:str,sample_key:str):return FileResponse(operation(registry().image,key,sample_key))


@router.post('/api/product-families/{key}/appearance/review')
def appearance_review(key:str,request:dict):return operation(registry().review,key,request)


def appearance_folder(key):
    registry().get(key)
    pointer=registry().folder(key)/'appearance_active.json'
    if not pointer.is_file():raise FileNotFoundError('外观匹配库尚未建立')
    binding=read(pointer);relative=binding['directory'];directory=(Path(registry().results)/relative).resolve()
    if Path(relative).is_absolute() or '..' in Path(relative).parts or not directory.is_relative_to((registry().results/'experiments').resolve()):
        raise ValueError('外观匹配版本路径无效')
    return directory


def get_appearance_model(key):
    directory=appearance_folder(key);meta_path=directory/'appearance.json';meta=read(meta_path)
    arrays=(directory/meta['arrays']).resolve()
    if arrays.parent!=directory:raise ValueError('外观参考文件路径无效')
    stamp=tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in (meta_path,arrays,directory/'report.json'))
    if stamp not in _appearance_cache:
        if meta['family_key']!=key:raise ValueError('外观库所属产品族不匹配')
        report=read(directory/'report.json')
        if report['arrays_sha256']!=meta['arrays_sha256'] or report['protocol_sha256']!=meta['protocol_sha256']:raise ValueError('外观模型与报告不配套')
        if meta['selected_method']=='clip':
            from web import server
            if file_hash(Path(server.ROOT)/'weights/CLIP-ViT-B-16-open_clip_model.safetensors')!=meta['weights_sha256']:
                raise ValueError('共享视觉编码器与外观参考版本不匹配')
        _appearance_cache.clear();_appearance_cache[stamp]=load_matcher(meta_path)
    return _appearance_cache[stamp]


@router.get('/api/product-families/{key}/matching')
def appearance_matching(key:str):
    operation(registry().get,key)
    try:
        model=get_appearance_model(key);report=read(model.path.parent/'report.json')
    except FileNotFoundError:return {'ok':True,'ready':False,'message':'外观匹配库正在准备；资料审核结果已独立保存'}
    except (KeyError,ValueError,OSError) as exc:raise HTTPException(503,'外观匹配库或报告校验失败') from exc
    return {'ok':True,'ready':True,'model':model.meta,'report':report,'quality_assessment':'not_available','automatic_archival':False}


def match_appearance(content,key):
    from web import server
    operation(registry().get,key)
    try:
        with Image.open(io.BytesIO(content)) as original:
            if original.format not in ('PNG','JPEG','WEBP','BMP') or original.width*original.height>20_000_000:raise ValueError('图片格式或尺寸不支持')
            image=original.convert('RGB')
        if min(image.size)<16 or max(image.size)<64 or np.asarray(image.convert('L')).std()<5:raise ValueError('图片过小或近乎空白，无法匹配外观')
    except (OSError,ValueError,Image.DecompressionBombError) as exc:raise HTTPException(400,str(exc)) from exc
    with server._inference_lock:
        try:model=get_appearance_model(key)
        except (OSError,ValueError,KeyError) as exc:raise HTTPException(503,'外观匹配库尚未就绪或校验失败') from exc
        if model.meta['selected_method']=='clip':
            detector,_=server.get_engine('screw');_,features=detector.extract_features(full_object(image))
        else:features=color_shape(image)
        result=model.predict(features)
        similarities=unit(np.asarray(features).reshape(-1))@unit(model.params[model.meta['selected_method']]).T
        nearest=np.argsort(-similarities)[:3];names={c['category_id']:c['name'] for c in model.meta['categories']}
        references=[{'key':model.meta['reference_keys'][i],'category_id':int(model.params['labels'][i]),
            'name':names[int(model.params['labels'][i])],'similarity':float(similarities[i]),
            'image_url':f"/api/product-families/{key}/appearance/{model.meta['reference_keys'][i]}/image"} for i in nearest]
        competitor=None
        if result.get('exclusion_reference_key'):
            key_id=result['exclusion_reference_key'];category=result['exclusion_category_id']
            competitor={'category_id':category,'name':next(c['name'] for c in model.meta['exclusion_categories'] if c['category_id']==category),
                'similarity':result['exclusion_similarity'],'provenance':'来源类别标注的排除参考，未作质量核验',
                'image_url':f'/api/product-families/{key}/appearance/{key_id}/image'}
    return {'ok':True,**result,'references':references,'exclusion_reference':competitor,'scope':'单件裁图的三类外观检索；不作质量判定，不自动定位，不自动归档'}


@router.post('/api/product-families/{key}/matching/detect')
async def appearance_match_detect(key:str,file:UploadFile=File(...)):
    content=await file.read(10*1024*1024+1)
    if not content or len(content)>10*1024*1024:raise HTTPException(400,'图片须小于10MB')
    return await run_in_threadpool(match_appearance,content,key)


def get_model(family='screw'):
    from web import server
    directory = folder(family)
    meta = read(directory / "family.json")
    files = [directory / "family.json", directory / meta["arrays"],directory/'report.json']
    stamp = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files)
    if stamp not in _model_cache:
        report=read(files[2])
        if report['arrays_sha256']!=meta['arrays_sha256'] or report['protocol_sha256']!=meta['protocol_sha256']:
            raise ValueError('产品族模型与报告不配套')
        if meta.get('family_key')!=family and not(family=='screw' and str(meta.get('version','')).startswith('screw-family-')):
            raise ValueError('模型所属产品族不匹配')
        model = FamilyModel(files[0])
        if file_hash(Path(server.ROOT) / "weights/CLIP-ViT-B-16-open_clip_model.safetensors") != meta["weights_sha256"]:
            raise ValueError("产品族模型与共享视觉编码器不匹配")
        _model_cache.clear()
        _model_cache[stamp] = model
    return _model_cache[stamp]


@router.get("/api/product-family/benchmark")
def benchmark(family:str='screw'):
    family_config=operation(registry().get,family)
    try:directory = folder(family)
    except FileNotFoundError:return {'ok':True,'ready':False,'family':family_config,'message':'该产品族已登记，尚无配套模型'}
    if not (directory / "report.json").is_file():
        return {"ok": True, "ready": False,'family':family_config, "message": "产品族对照实验尚未完成"}
    try:
        report = read(directory / "report.json")
        meta = read(directory / "family.json")
    except (OSError, ValueError):
        raise HTTPException(503, "产品族报告尚未就绪")
    if report["arrays_sha256"] != meta["arrays_sha256"] or report["protocol_sha256"] != meta["protocol_sha256"]:
        raise HTTPException(503, "产品族报告与模型不配套")
    return {"ok": True, "ready": True,'family':family_config, "report": report,
            "model": {"reference_groups": len(meta["groups"]), "unified_bank_files": 1,
                      "training_images": meta["training_images"], "calibration_images": meta["calibration_images"],
                      "training_views":meta.get('training_views',meta['training_images']),
                      "version":meta['version']},
            "advisory_only": True, "automatic_archival": False}


def preview(content,family='screw'):
    from web import server
    family_config=operation(registry().get,family)
    try:
        with Image.open(io.BytesIO(content)) as original:
            if original.format not in ("PNG", "JPEG", "WEBP", "BMP") or original.width * original.height > 20_000_000:
                raise ValueError("图片格式或尺寸不支持")
            image = original.convert("RGB")
        quality = image_quality(image)
        if not quality["ok"]: raise ValueError("图像质量不通过：" + "；".join(quality["reasons"]))
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, str(exc)) from exc
    with server._inference_lock:
        try:
            model = get_model(family)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(503, "产品族试验模型尚未就绪或校验失败") from exc
        encoder=family_config.get('encoder_product') or server.load_products()[0]['key']
        detector, _ = server.get_engine(encoder)
        patches, vector = detector.extract_features(image)
        result = model.predict(vector)
        raw = model.patch_scores(patches)
        group = result["routing"]["group"]
        n = int(np.sqrt(len(patches)))
        heatmap = detector._overlay(image, raw["grouped"][group].reshape(n, n))
    return {"ok": True,'family_key':family,'family_name':family_config['name'], **result, "quality_check": quality,
            "heatmap": server.img_to_b64(heatmap), "version": model.meta.get("version"),
            "conclusion": ("缺陷线索" if result["predictions"]["routed_expert"] == "defect" else "正常线索")
                          if result["routing"]["in_scope"] else "参考范围外，需复核",
            "decision_mode": "routed_expert",
            "scope": "工程回放试检；共用编码器与统一库，自动匹配轻量分组判别头；不识别具体型号；结果不归档、不修改模型或阈值"}


@router.post("/api/product-family/detect")
async def detect(file: UploadFile = File(...),family:str=Form('screw')):
    content = await file.read(10 * 1024 * 1024 + 1)
    if not content or len(content) > 10 * 1024 * 1024:
        raise HTTPException(400, "图片须小于10MB")
    return await run_in_threadpool(preview, content,family)
