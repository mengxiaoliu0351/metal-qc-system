"""Download a public teaching mirror, verify its published hash, then safely extract."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tarfile
import time
import threading

import requests

ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=ROOT/'data/external/mvtec_screws'


def main():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    session=requests.Session();session.trust_env=False
    response=session.get('https://api.osf.io/v2/files/ruca6/',timeout=30);response.raise_for_status()
    data=response.json()['data'];attributes=data['attributes']
    name=attributes['name'];expected=attributes['extra']['hashes']['sha256'];size=attributes['size']
    if name!='mvtec_screws_v1.0.tar.gz' or size!=1361662840:raise ValueError('Unexpected mirror version')
    source={'dataset':'MVTec Screws V1.0','publisher':'MVTec Software GmbH',
        'official':'https://www.mvtec.com/research-teaching/datasets/mvtec-screws',
        'mirror':'https://osf.io/ruca6/','mirror_context':'https://deeplearning.neuromatch.io/projects/ComputerVision/screws.html',
        'license':'CC BY-NC-SA 4.0','bytes':size,'sha256':expected,
        'label_scope':'object category and oriented bounding boxes; quality labels not supplied'}
    (DIRECTORY/'source.json').write_text(json.dumps(source,indent=2),encoding='utf8')
    archive=DIRECTORY/name;partial=archive.with_suffix(archive.suffix+'.part')
    if not archive.exists():
        parts=DIRECTORY/'download_parts';parts.mkdir(exist_ok=True);chunk_size=16*1024**2
        ranges=[(i,start,min(size,start+chunk_size)-1) for i,start in enumerate(range(0,size,chunk_size))]
        # Storage URLs expire after 60 seconds. Refresh centrally; never log them.
        resource=data['links'].get('move',data['links']['download'])
        url_lock=threading.Lock();url_state={'value':None,'time':0.}
        def resolve():
            with url_lock:
                if url_state['value'] and time.monotonic()-url_state['time']<25:return url_state['value']
                for attempt in range(6):
                    # GET on the advertised storage file resource is read-only.
                    with session.get(resource,params={'_':str(time.time_ns())},headers={'Range':'bytes=0-0','Cache-Control':'no-cache'},stream=True,timeout=(20,30)) as redirect:
                        if redirect.status_code==429:
                            time.sleep(min(60,max(10,int(redirect.headers.get('Retry-After','30')))));continue
                        if redirect.status_code!=206:raise RuntimeError('Public storage resolver returned HTTP '+str(redirect.status_code))
                        url_state.update(value=redirect.url,time=time.monotonic());return redirect.url
                raise RuntimeError('Public mirror remains throttled; completed parts retained')
        def download(part):
            index,start,end=part;path=parts/f'{index:04d}.bin';expected_size=end-start+1
            if path.exists() and path.stat().st_size==expected_size:return path
            client=requests.Session();client.trust_env=False
            for attempt in range(6):
                try:
                    completed=path.stat().st_size if path.exists() else 0
                    if completed%(2*1024**2) or completed>expected_size:completed=0
                    with path.open('ab' if completed else 'wb') as stream:
                        for begin in range(start+completed,end+1,2*1024**2):
                            finish=min(end,begin+2*1024**2-1)
                            r=client.get(resolve(),headers={'Range':f'bytes={begin}-{finish}'},timeout=(20,30))
                            if r.status_code==403:
                                with url_lock:url_state['time']=0.
                                raise ValueError('Public storage link expired; refresh before resuming')
                            if r.status_code==429:time.sleep(min(60,max(10,int(r.headers.get('Retry-After','30')))))
                            if r.status_code!=206:raise ValueError('Public storage returned HTTP '+str(r.status_code))
                            if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {begin}-{finish}/{size}' or len(r.content)!=finish-begin+1:raise ValueError('Invalid ranged response')
                            stream.write(r.content)
                    if path.stat().st_size!=expected_size:raise ValueError('Incomplete range')
                    return path
                except (requests.RequestException,ValueError):
                    if attempt==5:raise RuntimeError('Range download failed; completed segments retained') from None
            raise RuntimeError('Range download failed')
        began=time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i,path in enumerate(pool.map(download,ranges)):
                print(json.dumps({'parts_complete':i+1,'parts_total':len(ranges),'seconds':round(time.monotonic()-began)}),flush=True)
        with partial.open('wb') as stream:
            for i,_,_ in ranges:
                with (parts/f'{i:04d}.bin').open('rb') as source_stream:
                    for block in iter(lambda:source_stream.read(1024*1024),b''):stream.write(block)
        partial.replace(archive)
    digest=hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=expected:raise ValueError('Archive checksum mismatch')
    print('Archive verified against OSF SHA256',flush=True)
    destination=DIRECTORY/'source';marker=destination/'extracted.json'
    if marker.exists() and json.loads(marker.read_text())['sha256']==expected:return
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers()
        if sum(m.size for m in members)>8*1024**3:raise ValueError('Unexpected extracted size')
        for item in members:
            if not(item.isfile() or item.isdir()) or not (destination/item.name).resolve().is_relative_to(destination.resolve()):
                raise ValueError('Unsafe archive entry')
        bundle.extractall(destination,filter='data')
    marker.write_text(json.dumps({'sha256':expected,'members':len(members)}),encoding='utf8')
    print(json.dumps({'extracted':True,'members':len(members)}),flush=True)


if __name__=='__main__':main()
