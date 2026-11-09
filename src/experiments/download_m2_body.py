"""Resume the publisher's BODY archive with bounded ranges and SHA256 checks."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=ROOT/'data/external/m2_v2'


def main():
    metadata=json.loads((DIRECTORY/'source.json').read_text(encoding='utf8'))
    item=next(x for x in metadata['files'] if x['filename']=='BODY.rar')
    expected=item['content_details']['sha256_hash'];size=item['size'];archive=DIRECTORY/'BODY.rar'
    if size!=2919730041 or expected!='e660e1865310715451687db9657af26fd00da1f2fa69f91b10f5de80e17c4022':raise ValueError('Unexpected publisher archive version')
    if not archive.exists():
        session=requests.Session();session.trust_env=False
        with session.get(item['content_details']['download_url'],headers={'Range':'bytes=0-0'},stream=True,timeout=(20,30)) as response:
            if response.status_code!=206:raise ValueError('Publisher does not support expected range request')
            storage_url=response.url
        parts=DIRECTORY/'body_download_parts';parts.mkdir(exist_ok=True);chunk=16*1024**2
        ranges=[(i,start,min(start+chunk,size)-1) for i,start in enumerate(range(0,size,chunk))]
        def download(part):
            index,start,end=part;path=parts/f'{index:04d}.bin';length=end-start+1
            if path.exists() and path.stat().st_size==length:return index
            client=requests.Session();client.trust_env=False
            for attempt in range(5):
                try:
                    with client.get(storage_url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,45)) as r:
                        if r.status_code==429:time.sleep(min(60,int(r.headers.get('Retry-After','15'))))
                        if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise ValueError('Unexpected range response')
                        with path.open('wb') as stream:
                            for block in r.iter_content(128*1024):stream.write(block)
                    if path.stat().st_size!=length:raise ValueError('Incomplete range')
                    return index
                except (requests.RequestException,ValueError):
                    if attempt==4:raise RuntimeError('BODY range failed; completed chunks retained') from None
                    time.sleep(min(8,2**attempt))
        began=time.monotonic();done=0
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(download,part) for part in ranges]
            for future in as_completed(futures):
                future.result();done+=1
                if done%8==0 or done==len(ranges):print(json.dumps({'parts':done,'total':len(ranges),'seconds':round(time.monotonic()-began)}),flush=True)
        temporary=archive.with_suffix('.rar.part')
        with temporary.open('wb') as output:
            for index,_,_ in ranges:
                with (parts/f'{index:04d}.bin').open('rb') as stream:
                    for block in iter(lambda:stream.read(1024**2),b''):output.write(block)
        temporary.replace(archive)
    digest=hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):digest.update(chunk)
    if archive.stat().st_size!=size or digest.hexdigest()!=expected:raise ValueError('BODY archive does not match publisher hash')
    (DIRECTORY/'body-download-receipt.json').write_text(json.dumps({'verified':True,'filename':archive.name,'bytes':size,'sha256':expected,'source':metadata['url'],'license':metadata['license']},indent=2),encoding='utf8')
    print(json.dumps({'verified':True,'bytes':size,'sha256':expected}),flush=True)


if __name__=='__main__':main()
