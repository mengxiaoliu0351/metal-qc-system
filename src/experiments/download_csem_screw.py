"""Bounded, resumable download of the original CSEM screw archive only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import re
import threading
import time

import requests

from products.families import write

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT/'data/external/csem_misd'
URL = 'https://zenodo.org/records/7410513/files/screw.tar.gz?download=1'
SIZE = 6561524755
MD5 = '3c2c97d986be8475877ecff1b43ddc0d'
CHUNK = 16*1024**2


def download(directory, workers=6, minutes=25):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/'screw.tar.gz'; receipt = directory/'download_receipt.json'
    if target.exists() and receipt.exists() and __import__('json').loads(receipt.read_text())['verified_md5'] == MD5:
        print({'already_verified': True}, flush=True); return
    parts = directory/'download_parts'; parts.mkdir(exist_ok=True)
    started = time.monotonic(); deadline = started+minutes*60; stop = threading.Event()
    count = (SIZE+CHUNK-1)//CHUNK
    def current(): return sum(min(p.stat().st_size, CHUNK) for p in parts.glob('part-*.bin'))
    initial = current(); last = started
    def fetch(index):
        start = index*CHUNK; expected = min(CHUNK, SIZE-start); path = parts/f'part-{index:04d}.bin'
        session = requests.Session()
        for attempt in range(4):
            if stop.is_set() or time.monotonic() > deadline: raise TimeoutError('bounded download paused')
            length = path.stat().st_size if path.exists() else 0
            if length == expected: return index
            if length > expected: raise ValueError('Partial chunk exceeds expected length')
            offset = start+length; end = start+expected-1
            try:
                with session.get(URL, headers={'Range': f'bytes={offset}-{end}'}, stream=True, timeout=(12, 25)) as response:
                    if response.status_code in (429, 503):
                        time.sleep(min(30, int(response.headers.get('Retry-After', '10')))); continue
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{SIZE}':
                        raise ValueError('Server did not preserve the requested byte range')
                    with path.open('ab') as output:
                        for block in response.iter_content(1024*1024):
                            if stop.is_set() or time.monotonic() > deadline: raise TimeoutError('bounded download paused')
                            if output.tell()+len(block) > expected: raise ValueError('Response exceeds requested range')
                            output.write(block)
                if path.stat().st_size == expected: return index
            except requests.RequestException:
                if attempt == 3: raise
                time.sleep(2*(attempt+1))
        raise RuntimeError('Download chunk incomplete')
    error = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(fetch, i) for i in range(count)]
        for job in as_completed(jobs):
            try: job.result()
            except Exception as exc:
                error = type(exc).__name__; stop.set()
                for pending in jobs: pending.cancel()
                break
            now = time.monotonic()
            if now-last >= 20:
                total = current(); state = {'downloaded_bytes': total, 'total_bytes': SIZE, 'elapsed_seconds': round(now-started),
                    'new_MB_per_second': round((total-initial)/1024**2/max(now-started, 1), 2), 'complete': False}
                write(directory/'download_progress.json', state); print(state, flush=True); last = now
    if error:
        write(directory/'download_progress.json', {'downloaded_bytes': current(), 'total_bytes': SIZE,
              'complete': False, 'paused_reason': error, 'resume_supported': True})
        print({'complete': False, 'paused_reason': error, 'resume_supported': True}, flush=True); return
    temporary = directory/'screw.tar.gz.assembling'; md5 = hashlib.md5(); sha = hashlib.sha256(); size = 0
    with temporary.open('wb') as output:
        for i in range(count):
            with (parts/f'part-{i:04d}.bin').open('rb') as source:
                for block in iter(lambda: source.read(4*1024**2), b''):
                    md5.update(block); sha.update(block); size += len(block); output.write(block)
    if size != SIZE or md5.hexdigest() != MD5: raise ValueError('Original archive checksum mismatch; retain chunks for audit')
    temporary.replace(target)
    write(receipt, {'url': URL, 'bytes': size, 'verified_md5': md5.hexdigest(), 'sha256': sha.hexdigest(),
                    'source_record': 'https://zenodo.org/records/7410513', 'scope': 'original archive downloaded; no labels imported or models activated'})
    # Only remove the exact chunk files created by this downloader after the
    # complete original archive has passed the author's checksum.
    for i in range(count):
        path = (parts/f'part-{i:04d}.bin').resolve()
        if path.parent != parts.resolve(): raise ValueError('Unexpected chunk path')
        path.unlink()
    print({'complete': True, 'bytes': size, 'verified_md5': MD5}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--minutes', type=int, default=25)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.minutes <= 120: raise ValueError('Download budget out of range')
    download(args.directory.resolve(), args.workers, args.minutes)


if __name__ == '__main__': main()
