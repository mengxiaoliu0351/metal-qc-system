"""Read-only BODY quality research; never production archival or feedback labels."""
import io
from pathlib import Path
import re

import numpy as np
from PIL import Image
from fastapi import APIRouter,File,HTTPException,UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from perception.body_preprocess import prepare_body
from perception.body_quality import BodyQualityModel
from products.families import read
from review.workflow import file_hash

router=APIRouter();_cache={}


def directory(key):
    from web.family_api import registry,operation
    operation(registry().get,key)
    pointer=registry().folder(key)/'body_quality_candidate.json'
    if not pointer.is_file():raise FileNotFoundError('螺杆质量实验尚未完成')
    relative=read(pointer)['directory'];root=registry().results
    target=(root/relative).resolve()
    if Path(relative).is_absolute() or '..' in Path(relative).parts or not target.is_relative_to((root/'experiments').resolve()):raise ValueError('质量候选目录无效')
    return target


def model(key):
    from web import server
    folder=directory(key);files=[folder/name for name in ('quality.json','quality.npz','report.json','protocol.json')]
    runtime={name:Path(server.ROOT)/relative for name,relative in {
        'preprocessing_sha256':'src/perception/body_preprocess.py','extractor_sha256':'src/perception/winclip_patch.py'}.items()}
    stamp=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in files+list(runtime.values()))
    if stamp not in _cache:
        result=BodyQualityModel(files[0]);report=read(files[2])
        if result.meta['family_key']!=key or report['arrays_sha256']!=result.meta['arrays_sha256'] or report['protocol_sha256']!=result.meta['protocol_sha256']:raise ValueError('质量候选与报告不配套')
        protocol=read(files[3])
        if file_hash(files[3])!=result.meta['protocol_sha256'] or any(file_hash(path)!=protocol[field] for field,path in runtime.items()):raise ValueError('质量候选与冻结的区域处理或特征实现不配套')
        if file_hash(Path(server.ROOT)/'weights/CLIP-ViT-B-16-open_clip_model.safetensors')!=result.meta['weights_sha256']:raise ValueError('质量候选与编码器不配套')
        _cache.clear();_cache[stamp]=result
    return _cache[stamp]


@router.get('/api/product-families/{key}/body-quality')
def benchmark(key:str):
    try:
        folder=directory(key);result=model(key);report=read(folder/'report.json')
        examples=read(folder/'demo_samples.json')['samples'] if (folder/'demo_samples.json').exists() else []
        return {'ok':True,'ready':True,'trial_ready':report['eligible_for_research_trial'],'report':report,'model':result.meta,
            'examples':[{k:v for k,v in s.items() if k!='file'}|{'image_url':f"/api/product-families/{key}/body-quality/examples/{s['id']}"} for s in examples],
            'automatic_archival':False}
    except FileNotFoundError:return {'ok':True,'ready':False,'message':'螺杆质量数据与实验正在准备'}
    except (OSError,ValueError,KeyError) as exc:raise HTTPException(503,'螺杆质量候选校验失败') from exc


def example_path(key,sample_id):
    if not re.fullmatch('[a-f0-9]{64}',sample_id):raise HTTPException(400,'样例编号无效')
    try:
        folder=directory(key);index=read(folder/'demo_samples.json')
        sample=next((s for s in index['samples'] if s['id']==sample_id),None)
        if sample is None:raise HTTPException(404,'样例不存在')
        path=(folder/sample['file']).resolve()
        if not path.is_relative_to((folder/'demo_images').resolve()) or file_hash(path)!=sample['file_sha256']:raise ValueError('样例校验失败')
        return path
    except (FileNotFoundError,ValueError,KeyError) as exc:raise HTTPException(404,'样例尚未就绪') from exc


@router.get('/api/product-families/{key}/body-quality/examples/{sample_id}')
def example(key:str,sample_id:str):return FileResponse(example_path(key,sample_id))


def infer(content,key):
    from web import server
    try:
        with Image.open(io.BytesIO(content)) as original:
            if original.format not in ('PNG','JPEG','BMP','WEBP') or min(original.size)<128 or original.width*original.height>20_000_000:raise ValueError('请提供至少128像素、不超过2000万像素的单件图片')
            image=original.convert('RGB')
        if np.asarray(image.convert('L').resize((128,128))).std()<5:raise ValueError('图像近乎空白，无法进行质量研究试检')
    except (OSError,ValueError,Image.DecompressionBombError) as exc:raise HTTPException(400,str(exc)) from exc
    with server._inference_lock:
        try:quality_model=model(key);report=read(quality_model.path.parent/'report.json')
        except (OSError,KeyError,ValueError) as exc:raise HTTPException(503,'螺杆质量候选尚未就绪') from exc
        if not report['eligible_for_research_trial']:raise HTTPException(409,'候选未通过预设资格检查，保留报告但不启用试检')
        detector,_=server.get_engine('screw');prepared,region=prepare_body(image);patches,vector=detector.extract_features(prepared)
        local=quality_model.raw_scores(patches);result=quality_model.predict(vector,float(local.max()))
        overlay=detector._overlay(prepared,local.reshape(14,14))
    return {'ok':True,**result,'input_region':region,'heatmap':server.img_to_b64(overlay),'heatmap_scope':'候选裁剪及补边后的外观差异线索；未验证框或像素定位准确率',
        'scope':'M2螺杆公开数据下的本地质量研究；不调用VLM，不归档，不生成反馈标签，最终需人工核验'}


@router.post('/api/product-families/{key}/body-quality/detect')
async def detect(key:str,file:UploadFile=File(...)):
    content=await file.read(10*1024*1024+1)
    if not content or len(content)>10*1024*1024:raise HTTPException(400,'每张图片须小于10MB')
    return await run_in_threadpool(infer,content,key)


@router.post('/api/product-families/{key}/body-quality/examples/{sample_id}/detect')
async def example_detect(key:str,sample_id:str):
    content=await run_in_threadpool(example_path,key,sample_id)
    return await run_in_threadpool(infer,content.read_bytes(),key)
