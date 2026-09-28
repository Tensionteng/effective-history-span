#!/usr/bin/env python
"""Decay real-data recalibration (E9): estimate c_real from real-data grid optima.

Motivation: the synthetic-drift constant c=65.63 puts every real dataset's raw N*
(242-355) below the floor L/3=960 at sl=2880, so decay-auto degenerates to a single
universal mu=0.998959 (logs/decay/REPORT.md, floor section). Recalibration chain:
  drift (train split, as printed by the existing auto logs — the exact values the
  resolver saw) -> lambda via LOG-LOG EXTRAPOLATION of logs/power_law _curve
  (np.polyfit on log-log; replaces endpoint clamping, which is what destroyed the
  drift resolution for exchange/ETTh1) -> per-dataset c_i = N*_opt * lam^(1/3), where
  N*_opt = grid-argmin effective window of the mu scan (iTransformer, logs/decay;
  weather's optimum is pinned by the mu{0.996,0.998} mini-grid plus the archived
  no-floor auto arm N*=287) -> c_real = geomean(c_i) = least-squares refit of the
  intercept at fixed slope -1/3 (same estimator that produced c=65.63 on synthetic).

Phases:
  estimate  print/write the calibration table + predicted per-dataset mu
            (logs/decay/recalib.json)
  collect   after the autorc validation arms: verdict table (mu distinct? <= none?)

Usage: .venv/bin/python analysis/decay_recalib.py estimate|collect
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_provider.data_factory import _drift_to_lambda, POWER_LAW_C

LOGDIR = 'logs/decay'
EHS_V2 = 'logs/ehs_v2'
OUT_JSON = 'logs/decay/recalib.json'
SEEDS = [2021, 2022]
DATASETS = ['exchange_rate', 'ETTh1', 'weather', 'electricity']
GRID_MU = {0.995: 'mu0p995', 0.996: 'mu0p996', 0.998: 'mu0p998', 0.999: 'mu0p999',
           0.9995: 'mu0p9995'}

re_mse = re.compile(r'^mse:([\d.eE+-]+)', re.M)
re_auto = re.compile(r'decay_auto: drift=([\d.]+) -> lambda=([\d.]+) -> N\*=(\d+) -> mu=([\d.]+)')


def log_mse(path):
    if not os.path.exists(path):
        return None
    txt = open(path, errors='ignore').read()
    m = re_mse.findall(txt)
    return float(m[-1]) if m else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    return float(np.std(xs)) if len(xs) > 1 else 0.0


def arm_mses(dataset, model, tag, logdir=LOGDIR, seeds=SEEDS):
    return [log_mse(os.path.join(logdir, f'{dataset}_{model}_{tag}_sl2880_s{s}.log'))
            for s in seeds]


def parse_drifts():
    """drift/lambda/raw-N* from the archived no-floor auto logs (values the resolver saw)."""
    drifts = {}
    ddir = os.path.join(LOGDIR, 'v1_nofloor')
    for fn in sorted(os.listdir(ddir)):
        m = re.match(r'([A-Za-z0-9_]+?)_(iTransformer|PatchTSTGated)_auto_sl2880_s\d+\.log', fn)
        if not m:
            continue
        txt = open(os.path.join(ddir, fn), errors='ignore').read()
        a = re_auto.search(txt)
        if a and m.group(1) not in drifts:
            drifts[m.group(1)] = {'drift': float(a.group(1)),
                                  'lam_clamped': float(a.group(2)),
                                  'nstar_raw_synth': int(a.group(3))}
    return drifts


def grid_table(dataset, model):
    """mean test mse per mu over the available grid arms + auto variants + none."""
    tab = {}
    for mu, tag in GRID_MU.items():
        v = arm_mses(dataset, model, tag)
        if any(x is not None for x in v):
            tab[-1.0 / np.log(mu)] = v
    # no-floor auto (archived) and floor auto
    v = arm_mses(dataset, model, 'auto', logdir=os.path.join(LOGDIR, 'v1_nofloor'))
    if any(x is not None for x in v):
        txt = open(os.path.join(LOGDIR, 'v1_nofloor',
                                f'{dataset}_{model}_auto_sl2880_s2021.log'), errors='ignore').read()
        a = re_auto.search(txt)
        tab[int(a.group(3))] = v
    v = arm_mses(dataset, model, 'auto')
    if any(x is not None for x in v):
        tab[960] = v  # floor-saturated auto: N*=L/3=960
    v = arm_mses(dataset, model, 'none')
    if any(x is not None for x in v):
        tab[np.inf] = v
    return tab


def estimate():
    drifts = parse_drifts()
    # log-log extrapolation coefficients (for the record)
    curve = json.load(open('logs/power_law/calibration.json'))['_curve']
    pts = sorted((v['drift_mean'], v['lam']) for v in curve.values())
    b, a = np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)

    est = {}
    for ds in DATASETS:
        d = drifts[ds]['drift']
        lam_ext = _drift_to_lambda(d, extrapolate=True)
        tab = grid_table(ds, 'iTransformer')
        means = {n: mean(v) for n, v in tab.items()}
        n_opt = min(means, key=means.get)
        c_i = n_opt * lam_ext ** (1.0 / 3.0) if np.isfinite(n_opt) else None
        est[ds] = {'drift': d, 'lam_ext': lam_ext, 'lam_clamped': drifts[ds]['lam_clamped'],
                   'grid_mean_mse': {str(k): v for k, v in sorted(means.items())},
                   'nstar_opt': (n_opt if np.isfinite(n_opt) else 'inf'),
                   'c_i': c_i}
    cvals = [e['c_i'] for e in est.values() if e['c_i']]
    c_real = float(np.exp(np.mean(np.log(cvals)))) if len(cvals) == len(DATASETS) else None
    c_3pt = float(np.exp(np.mean(np.log([e['c_i'] for e in est.values()
                                         if e['c_i'] and e['nstar_opt'] != 'inf'
                                         and e['grid_mean_mse']])))) if cvals else None
    # sensitivity: 3-point (without weather) and +PTG exchange point
    c_no_weather = float(np.exp(np.mean(np.log([est[ds]['c_i'] for ds in
                                                ('exchange_rate', 'ETTh1', 'electricity')
                                                if est[ds]['c_i']]))))
    ptg = grid_table('exchange_rate', 'PatchTSTGated')
    ptg_means = {n: mean(v) for n, v in ptg.items()}
    ptg_opt = min(ptg_means, key=ptg_means.get)
    c_ptg = ptg_opt * est['exchange_rate']['lam_ext'] ** (1.0 / 3.0)
    c_with_ptg = float(np.exp(np.mean(np.log(cvals + [c_ptg])))) if c_real else None

    pred = {}
    for ds in DATASETS:
        lam = est[ds]['lam_ext']
        for tag, c in [('c_real', c_real), ('c_no_weather', c_no_weather),
                       ('c_synth', POWER_LAW_C)]:
            if c:
                nstar = c * lam ** (-1.0 / 3.0)
                pred.setdefault(ds, {})[tag] = {'c': c, 'nstar': nstar,
                                                'mu': float(np.exp(-1.0 / nstar))}

    out = {'fit': {'loglog_slope': float(b), 'loglog_intercept': float(a)},
           'per_dataset': est, 'c_real_4pt': c_real, 'c_no_weather_3pt': c_no_weather,
           'c_ptg_exchange': float(c_ptg), 'c_with_ptg_5pt': c_with_ptg,
           'predicted_auto': pred}
    return out


def collect(c_used):
    rows = []
    for ds in DATASETS:
        mses = arm_mses(ds, 'iTransformer', 'autorc')
        txt_path = os.path.join(LOGDIR, f'{ds}_iTransformer_autorc_sl2880_s2021.log')
        mu_res = nstar_res = drift_res = lam_res = None
        if os.path.exists(txt_path):
            a = re_auto.search(open(txt_path, errors='ignore').read())
            if a:
                drift_res, lam_res, nstar_res, mu_res = (float(a.group(1)), float(a.group(2)),
                                                         int(a.group(3)), float(a.group(4)))
        none = arm_mses(ds, 'iTransformer', 'none')
        if not any(x is not None for x in none):  # electricity/ETTh1/weather: ehs_v2 anchors
            none = arm_mses(ds, 'iTransformer', 'sl2880', logdir=EHS_V2)
            # ehs_v2 naming: {ds}_iTransformer_sl2880_s{seed}.log
            none = [log_mse(os.path.join(EHS_V2, f'{ds}_iTransformer_sl2880_s{s}.log'))
                    for s in SEEDS]
        floor_auto = arm_mses(ds, 'iTransformer', 'auto')
        rows.append({'dataset': ds, 'drift': drift_res, 'lam_ext': lam_res,
                     'nstar': nstar_res, 'mu': mu_res,
                     'autorc_mse': mses, 'autorc_mean': mean(mses), 'autorc_std': std(mses),
                     'none_mse': none, 'none_mean': mean(none),
                     'floor_auto_mean': mean(floor_auto)})
    mus = [r['mu'] for r in rows if r['mu']]
    distinct = len(set(mus)) == len(mus) and mus
    not_worse = all(r['autorc_mean'] is not None and r['none_mean'] is not None and
                    r['autorc_mean'] <= r['none_mean'] for r in rows)
    return {'c_used': c_used, 'rows': rows, 'mu_all_distinct': bool(distinct),
            'not_worse_than_none_all': bool(not_worse)}


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else 'estimate'
    c_used = float(sys.argv[2]) if len(sys.argv) > 2 else None
    est = estimate()
    if phase in ('estimate', 'all'):
        f = est['fit']
        print(f"log-log extrapolation fit: log lam = {f['loglog_slope']:.4f} * log drift "
              f"{f['loglog_intercept']:+.4f}")
        print(f"{'dataset':15} {'drift':>7} {'lam_clamp':>10} {'lam_ext':>9} "
              f"{'N*_opt':>7} {'c_i':>8}")
        for ds in DATASETS:
            e = est['per_dataset'][ds]
            print(f"{ds:15} {e['drift']:7.4f} {e['lam_clamped']:10.6f} {e['lam_ext']:9.6f} "
                  f"{str(e['nstar_opt']):>7} {e['c_i'] if e['c_i'] else float('nan'):8.2f}")
        print(f"\nc_real (4-pt geomean) = {est['c_real_4pt']}")
        print(f"c sensitivity: 3-pt (no weather) = {est['c_no_weather_3pt']:.2f}, "
              f"+PTG-exchange 5-pt = {est['c_with_ptg_5pt']}, "
              f"PTG-exchange c_i = {est['c_ptg_exchange']:.2f}, synthetic c = {POWER_LAW_C}")
        if est['c_real_4pt']:
            print('\npredicted recalibrated auto (no floor):')
            for ds in DATASETS:
                p = est['predicted_auto'][ds]['c_real']
                print(f"  {ds:15} N*={p['nstar']:7.0f}  mu={p['mu']:.6f}")
    out = {'estimate': est}
    if phase in ('collect', 'all'):
        if c_used is None:
            c_used = est['c_real_4pt']
        col = collect(c_used)
        out['collect'] = col
        print(f"\n== autorc validation (c={c_used}) ==")
        print(f"{'dataset':15} {'mu':>9} {'N*':>6} {'autorc':>18} {'none':>8} "
              f"{'floor-auto':>10} {'<=none':>7}")
        for r in col['rows']:
            fa = f"{r['floor_auto_mean']:.4f}" if r['floor_auto_mean'] else '-'
            ok = (r['autorc_mean'] is not None and r['none_mean'] is not None
                  and r['autorc_mean'] <= r['none_mean'])
            print(f"{r['dataset']:15} {r['mu']:9.6f} {r['nstar']:6} "
                  f"{r['autorc_mean']:.4f}±{r['autorc_std']:.4f}  {r['none_mean']:8.4f} "
                  f"{fa:>10} {'PASS' if ok else 'FAIL':>7}")
        print(f"mu all distinct: {col['mu_all_distinct']}; "
              f"not worse than none on all: {col['not_worse_than_none_all']}")
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print(f'\nwrote {OUT_JSON}')


if __name__ == '__main__':
    main()
