#!/usr/bin/env python
"""Download all data-*.arrow files of Salesforce/GiftEval via hf-mirror.com."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

RAW = 'logs/gift_ext/data/raw'
BASE = 'https://hf-mirror.com/datasets/Salesforce/GiftEval/resolve/main/'

tree = json.load(open('logs/gift_ext/tree.json'))
files = sorted(x['path'] for x in tree
               if x['path'].endswith('.arrow') and '/data-' in x['path'])


def fetch(path):
    out = os.path.join(RAW, path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return path, 'cached'
    for attempt in range(4):
        r = subprocess.run(['curl', '-sL', '--max-time', '600', '-o', out, BASE + path],
                           capture_output=True)
        if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 1000:
            return path, f'ok {os.path.getsize(out)/1e6:.1f}MB'
    return path, 'FAILED'


with ThreadPoolExecutor(max_workers=8) as ex:
    for path, status in ex.map(fetch, files):
        print(f'{status:>12}  {path}', flush=True)
print('DONE', file=sys.stderr)
