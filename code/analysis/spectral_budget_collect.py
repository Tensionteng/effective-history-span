#!/usr/bin/env python
"""Collect spectral-budget arm metrics from logs/spectral_budget/ and emit verdict tables.

Parses the final 'mse:..., mae:...' line of each arm log, aggregates over seeds
(mean±std), and prints the D1-D4 verdict tables plus cross-reference checks
against logs/ehs_v2 (same-protocol none arms) and logs/ehs_fix/e4 (PatchTSTGated
hard-budget protocol, cited as reference only).

Run: .venv/bin/python analysis/spectral_budget_collect.py
"""
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOGDIR = os.path.join(ROOT, 'logs', 'spectral_budget')

# ehs_v2 main-matrix references (same protocol: max_train_windows=N_min, seeds 2021/2022)
EHSV2_REF = {
    ('electricity', 'iTransformer', 2880): (0.1307, 0.0007, 2),
    ('electricity', 'DLinear', 2880): (0.1291, 0.0, 1),
    ('exchange_rate', 'iTransformer', 2880): (0.3211, 0.0119, 2),
    ('exchange_rate', 'DLinear', 2880): (0.6356, 0.0109, 2),
}
# E4 references (PatchTSTGated, UNCAPPED windows; different protocol -> reference only)
E4_REF = {
    ('electricity', 1440): {'zero': 0.1283, 'none': 0.1301, 'native': 0.1291},
    ('exchange_rate', 96): {'zero': 0.1013, 'none': 0.3989, 'native': 0.0950},
}

LOG_RE = re.compile(r'^(electricity|exchange_rate)_(iTransformer|DLinear|TimesNet)_'
                    r'(none|zero\d+|level\d+|spectral\d+|downsample\d+)_sl(\d+)_s(\d+)\.log$')
MSE_RE = re.compile(r'mse:([0-9.]+), mae:([0-9.]+)')


def collect():
    res = defaultdict(dict)  # (dataset, model, tag, sl) -> {seed: (mse, mae)}
    for fn in sorted(os.listdir(LOGDIR)):
        m = LOG_RE.match(fn)
        if not m:
            continue
        dataset, model, tag, sl, seed = m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5))
        path = os.path.join(LOGDIR, fn)
        found = None
        with open(path, 'rb') as f:
            f.seek(max(0, os.path.getsize(path) - 4000))
            tail = f.read().decode(errors='ignore')
        for mm in MSE_RE.finditer(tail):
            found = (float(mm.group(1)), float(mm.group(2)))
        if found is None:
            print(f'  [incomplete] {fn}', file=sys.stderr)
            continue
        res[(dataset, model, tag, sl)][seed] = found
    return res


def fmt(vals):
    if not vals:
        return '-'
    ms = [v[0] for v in vals.values()]
    mean = sum(ms) / len(ms)
    if len(ms) > 1:
        var = sum((x - mean) ** 2 for x in ms) / (len(ms) - 1)
        return f'{mean:.4f}±{var ** 0.5:.4f}[n={len(ms)}]'
    return f'{mean:.4f}[n=1]'


def main():
    res = collect()
    print(f'collected {sum(len(v) for v in res.values())} finished arms\n')

    def cell(dataset, model, tag, sl):
        return fmt(res.get((dataset, model, tag, sl), {}))

    for dataset, budget, sl in (('electricity', 1440, 2880), ('exchange_rate', 96, 2880)):
        print(f'### {"D1" if dataset == "electricity" else "D2"}: {dataset} sl={sl} b={budget}')
        print('| model | none | zero | level | spectral | downsample |')
        print('|---|---|---|---|---|---|')
        for model in ('iTransformer', 'DLinear'):
            print(f'| {model} | {cell(dataset, model, "none", sl)} | '
                  f'{cell(dataset, model, f"zero{budget}", sl)} | '
                  f'{cell(dataset, model, f"level{budget}", sl)} | '
                  f'{cell(dataset, model, f"spectral{budget}", sl)} | '
                  f'{cell(dataset, model, f"downsample{budget}", sl)} |')
        ref = EHSV2_REF.get((dataset, 'iTransformer', sl))
        print(f'ehs_v2 none ref: iTransformer {ref[0]}±{ref[1]}[n={ref[2]}], '
              f'DLinear {EHSV2_REF[(dataset, "DLinear", sl)]}', )
        e4 = E4_REF[(dataset, budget)]
        print(f'E4 ref (PatchTSTGated, uncapped): zero={e4["zero"]}, none={e4["none"]}, native sl=b={e4["native"]}\n')

    print(f'### D3: electricity TimesNet sl=1440 (sl=2880 probe 3.86s/it -> ~5.1h/arm > 4h, fallback)')
    print('| variant | mse |')
    print('|---|---|')
    for tag in ('none', 'spectral720'):
        print(f'| {tag} | {cell("electricity", "TimesNet", tag, 1440)} |')
    print('ehs_v2 TimesNet electricity refs [n=1]: sl96=0.1687, sl336=0.1794, sl720=0.1895 '
          '(sl1440/2880 never finished in ehs_v2: OOM under contention)')


if __name__ == '__main__':
    main()
