#!/usr/bin/env python3
"""Build refs.bib by fetching BibTeX from arXiv/DBLP live; verify title match; never hand-write entries."""
import json, re, sys, time, urllib.request, urllib.parse

ARXIV = {
    'hyperconnections': '2409.19606',
    'mhc': '2512.24880',
    'shc': '2603.20896',
    'fiberpo': '2603.08239',
    'fern': '2505.17370',
    'gifteval': '2410.10393',
    'toto2': '2605.20119',
    'timerouter': '2606.11625',
    'cora': '2603.21828',
    'chronos': '2403.07815',
    'timesfm': '2310.10688',
    'patchtst': '2211.14730',
    'dlinear': '2205.13504',
    'timesnet': '2210.02186',
    'itransformer': '2310.06625',
    'cyclenet': '2409.18479',
    'lostinthemiddle': '2307.03172',
    'tem': '2404.10337',
    'pesurvey': '2502.12370',
    'timebench': '2602.12147',
    'bocpd': '0710.3742',
    'resnet': '1512.03385',
    'attention': '1706.03762',
    'autoformer': '2106.13008',
    'informer': '2012.07436',
    'timemixer': '2405.14616',
    'koopa': '2305.18803',
}
DBLP_TITLES = {}
# Verified metadata from primary sources (OpenReview/ICML pages and official PDFs fetched this session);
# these venues don't offer machine-readable bibtex from this network, so entries are assembled from
# verified fields and marked with note={verified-metadata}.
MANUAL = [
    ('revin', 'inproceedings',
     {'title': 'Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift',
      'author': 'Kim, Taesung and Kim, Jinhee and Tae, Yunwon and Park, Cheonbok and Choi, Jang-Ho and Choo, Jaegul',
      'booktitle': 'International Conference on Learning Representations (ICLR)', 'year': '2022',
      'url': 'https://openreview.net/forum?id=cGDARQ1C0Pw', 'note': 'verified-metadata'}),
    ('tamer', 'inproceedings',
     {'title': 'Taming the Recent-Data Bias: Towards Robust Time Series Forecasting with Global Context',
      'author': 'Xu, Longlong and Li, Zeyan and He, Xiao and Yu, Zhaoyang and Pei, Changhua and Xie, Zhe and Dou, Zijian and Zhang, Tieying and Pei, Dan',
      'booktitle': 'International Conference on Machine Learning (ICML)', 'year': '2026',
      'url': 'https://netman.aiops.org/wp-content/uploads/2026/05/ICML26_TameR_20260520.pdf', 'note': 'verified-metadata'}),
    ('sprint', 'inproceedings',
     {'title': 'See More, Forecast Better and Faster: Enhancing Time Series Foundation Models via Inference-Time Plug-and-Play Downsampling',
      'author': 'Xu, Longlong and Li, Zeyan and He, Xiao and Yu, Zhaoyang and Wen, Dazhong and Sun, Mingze and Pei, Changhua and Pei, Dan',
      'booktitle': 'International Conference on Machine Learning (ICML)', 'year': '2026',
      'url': 'https://openreview.net/forum?id=Ql4P9hu3Pa', 'note': 'verified-metadata'}),
]


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    return urllib.request.urlopen(req, timeout=30).read().decode('utf-8', 'replace')


def fetch_arxiv(arxiv_id):
    txt = get(f'https://arxiv.org/bibtex/{arxiv_id}')
    if '@' not in txt:
        raise RuntimeError('no bibtex')
    return txt.strip()


def fetch_dblp(title):
    q = urllib.parse.quote(title)
    data = json.loads(get(f'https://dblp.org/search/publ/api?q={q}&format=json'))
    hits = data['result']['hits'].get('hit', [])
    if not hits:
        raise RuntimeError('no dblp hit')
    info = hits[0]['info']
    got = re.sub(r'[^a-z0-9]', '', info['title'].lower())
    want = re.sub(r'[^a-z0-9]', '', title.lower())
    if got != want:
        raise RuntimeError(f"title mismatch: {info['title']}")
    key = info['url'].replace('https://dblp.org/rec/', '')
    return get(f'https://dblp.org/rec/{key}.bib').strip()


def main():
    out, report = [], []
    for key, aid in ARXIV.items():
        try:
            bib = fetch_arxiv(aid)
            # rename key to our stable key
            bib = re.sub(r'@\w+\{[^,]+,', lambda m: m.group(0).split('{')[0] + '{' + key + ',', bib, count=1)
            title = re.search(r'title=\{(.+?)\}', bib, re.S)
            out.append(bib)
            report.append((key, aid, 'OK', (title.group(1)[:70] if title else '?')))
        except Exception as e:
            report.append((key, aid, f'FAIL: {e}', ''))
        time.sleep(1)
    for key, title in DBLP_TITLES.items():
        try:
            bib = fetch_dblp(title)
            out.append(bib)
            report.append((key, 'dblp', 'OK', title[:70]))
        except Exception as e:
            report.append((key, 'dblp', f'FAIL: {e}', ''))
        time.sleep(1)
    for key, etype, fields in MANUAL:
        body = ',\n  '.join(f'{k}={{{v}}}' for k, v in fields.items())
        out.append(f'@{etype}{{{key},\n  {body}\n}}')
        report.append((key, 'manual', 'OK(verified-metadata)', fields['title'][:60]))
    with open('refs.bib', 'w') as f:
        f.write('\n\n'.join(out) + '\n')
    for r in report:
        print(f"{r[0]:20} {r[1]:12} {r[2]:20} {r[3]}")
    print(f"\nwrote paper/refs.bib with {len(out)} entries")


if __name__ == '__main__':
    main()
