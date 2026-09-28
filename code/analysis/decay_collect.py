#!/usr/bin/env python
"""Collect decay-experiment arms from logs/decay/ and print verdict tables.

References cited (protocol-tagged):
  ehs_v2 main matrix (capped N_min, seeds 2021/2022, same configs as here):
    iTransformer none sl2880: exchange 0.3211, electricity 0.1307, ETTh1 0.4527, weather 0.2244
    iTransformer native best: exchange sl96 0.0889, ETTh1 sl720 0.3915, weather sl336 0.1632
  spectral_budget (capped, iTransformer): exchange zero96 0.5999; electricity zero1440 0.1291
  E4 (PatchTSTGated, UNCAPPED): exchange none 0.3989, zero96 0.1013, native sl96 0.0950;
    ETTh1 native sl720 0.4204; weather native sl336 0.1528

Run: .venv/bin/python analysis/decay_collect.py
"""
import os
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOGDIR = os.path.join(ROOT, 'logs', 'decay')
MU_ORDER = ['mu0p995', 'mu0p998', 'mu0p999', 'mu0p9995', 'auto', 'none', 'zero96']
MU_LABEL = {'mu0p995': 'μ=0.995 (N*≈200)', 'mu0p998': 'μ=0.998 (N*≈500)',
            'mu0p999': 'μ=0.999 (N*≈1000)', 'mu0p9995': 'μ=0.9995 (N*≈2000)',
            'auto': 'auto', 'none': 'none (μ=1)', 'zero96': 'zero b=96'}

LOG_RE = re.compile(r'^(exchange_rate|electricity|ETTh1|weather)_(iTransformer|PatchTSTGated)_'
                    r'(mu\dp\d+|auto|none|zero96)_sl2880_s(\d+)\.log$')
MSE_RE = re.compile(r'mse:([0-9.]+), mae:([0-9.]+)')
AUTO_RE = re.compile(r'decay_auto: drift=([0-9.]+) -> lambda=([0-9.]+) -> N\*=([0-9]+) -> mu=([0-9.]+)')

REFS = {
    ('exchange_rate', 'iTransformer'): dict(none=0.3211, native=0.0889, zero=0.5999,
                                            native_src='ehs_v2 sl96', zero_src='spectral_budget capped'),
    ('exchange_rate', 'PatchTSTGated'): dict(none=0.3989, native=0.0950, zero=0.1013,
                                             native_src='E4 uncapped', zero_src='E4 uncapped'),
    ('electricity', 'iTransformer'): dict(none=0.1307, native=0.1307, zero=0.1291,
                                          native_src='ehs_v2 sl2880(best)', zero_src='spectral_budget capped b=1440'),
    ('ETTh1', 'iTransformer'): dict(none=0.4527, native=0.3915, zero=None,
                                    native_src='ehs_v2 sl720(best)', zero_src='-'),
    ('weather', 'iTransformer'): dict(none=0.2244, native=0.1632, zero=None,
                                      native_src='ehs_v2 sl336(best)', zero_src='-'),
}
E4_NATIVE = {'ETTh1': 0.4204, 'weather': 0.1528}  # PatchTSTGated uncapped native, task-cited


def collect():
    res = defaultdict(dict)
    autos = {}
    for fn in sorted(os.listdir(LOGDIR)):
        m = LOG_RE.match(fn)
        if not m:
            continue
        ds, model, tag, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
        path = os.path.join(LOGDIR, fn)
        with open(path, 'rb') as f:
            tail = f.read().decode(errors='ignore')
        found = None
        for mm in MSE_RE.finditer(tail):
            found = (float(mm.group(1)), float(mm.group(2)))
        if found is None:
            print(f'  [incomplete] {fn}', file=sys.stderr)
            continue
        res[(ds, model, tag)][seed] = found
        if tag == 'auto' and (ds, model) not in autos:
            am = AUTO_RE.search(tail)
            if am:
                autos[(ds, model)] = am.groups()
    return res, autos


def fmt(res, ds, model, tag):
    vals = res.get((ds, model, tag), {})
    if not vals:
        return '-'
    ms = [v[0] for v in vals.values()]
    mean = float(np.mean(ms))
    if len(ms) > 1:
        return f'{mean:.4f}±{float(np.std(ms, ddof=1)):.4f}[n={len(ms)}]'
    return f'{mean:.4f}[n=1]'


def main():
    res, autos = collect()
    print(f'collected {sum(len(v) for v in res.values())} finished arms\n')

    combos = [('exchange_rate', 'iTransformer'), ('exchange_rate', 'PatchTSTGated'),
              ('electricity', 'iTransformer'), ('ETTh1', 'iTransformer'), ('weather', 'iTransformer')]
    for ds, model in combos:
        ref = REFS[(ds, model)]
        print(f'### {ds} × {model} (sl=2880)')
        print('| arm | test MSE |')
        print('|---|---|')
        best_grid, best_mu = None, None
        for tag in MU_ORDER:
            v = fmt(res, ds, model, tag)
            if v == '-':
                continue
            print(f'| {MU_LABEL[tag]} | {v} |')
            if tag.startswith('mu'):
                mean = float(np.mean([x[0] for x in res[(ds, model, tag)].values()]))
                if best_grid is None or mean < best_grid:
                    best_grid, best_mu = mean, tag
        if (ds, model) in autos:
            d, lam, ns, mu = autos[(ds, model)]
            print(f'  auto resolved: drift={d}, lambda={lam}, N*={ns}, mu={mu}')
        if best_mu:
            print(f'  best grid: {MU_LABEL[best_mu]} -> {best_grid:.4f}')
        print(f'  refs: none={ref["none"]} ({ref.get("native_src", "")}), '
              f'native-best={ref["native"]}, zero={ref["zero"]}')
        print()


if __name__ == '__main__':
    main()
