#!/usr/bin/env python
"""E7: BOCPD falsification robustness pack (review ccfa-review-reports/ehs-review.md E7).

E7.1 multiple-comparison correction of the exp4 hazard scan: exact permutation
p-values (n=8, full 8! enumeration) + Bonferroni/Holm over the 5-hazard x 2-model
family (m=10) and per-model families (m=5). Input: the published exp4 table
(logs/mechanism/exp4/bocpd_summary.md, integers; verified that ranks coincide
with the unrounded values, so rho and the permutation p are unaffected).

E7.2 deseasonalized BOCPD re-run: 'phase' (subtract per-phase means at the
calendar period) and 'diff' (seasonal difference at the same lag) variants.
Worker mode computes one (variant, dataset); merge mode builds the tables and
Spearman direction verdict for electricity/traffic.

E7.3 BOCPD-free drift statistics vs measured best lookback: rolling mean-step
and rolling symmetrized Gaussian KL (both changepoint-model-free), Spearman with
exact permutation p.

Usage:
  python analysis/e7_bocpd_robust.py mult                 # E7.1
  python analysis/e7_bocpd_robust.py deseas-worker --variant phase --dataset weather
  python analysis/e7_bocpd_robust.py deseas-merge         # E7.2 tables
  python analysis/e7_bocpd_robust.py drift                # E7.3
Outputs: logs/ehs_fix/e7/
"""
import argparse
import json
import os
import sys
import time
from itertools import permutations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bocpd_ehs import (ROOT, DATASETS, BEST_SL, HAZARDS, R_MAX, BURN_IN,
                       bocpd_runlength, rankdata, spearman)

OUTD = os.path.join(ROOT, 'logs', 'ehs_fix', 'e7')
os.makedirs(OUTD, exist_ok=True)

PERIODS = {'ETTh1': 24, 'ETTh2': 24, 'ETTm1': 96, 'ETTm2': 96,
           'exchange_rate': 7, 'weather': 144, 'electricity': 24, 'traffic': 24}

# published exp4 table (logs/mechanism/exp4/bocpd_summary.md); order = HAZARDS
ER_ORIG = {
    'ETTh1': [65, 76, 90, 102, 114],
    'ETTh2': [234, 248, 262, 271, 279],
    'ETTm1': [60, 64, 68, 71, 74],
    'ETTm2': [363, 377, 390, 399, 407],
    'exchange_rate': [210, 215, 219, 221, 223],
    'weather': [291, 297, 301, 303, 305],
    'electricity': [53, 68, 122, 234, 372],
    'traffic': [48, 59, 87, 146, 238],
}


def rho_of(rx, ry):
    rx = np.asarray(rx, float) - np.mean(rx)
    ry = np.asarray(ry, float) - np.mean(ry)
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def spearman_exact_p(x, y):
    """Spearman rho + exact two-sided permutation p (enumerates all n! orderings
    of y, ties included with multiplicity). Feasible for n<=9."""
    rx = rankdata(x)
    ry = rankdata(y)
    obs = rho_of(rx, ry)
    n = len(ry)
    cnt, tot = 0, 0
    idx = np.arange(n)
    for perm in permutations(idx):
        r = rho_of(rx, ry[list(perm)])
        tot += 1
        if abs(r) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, cnt / tot


def holm_adjust(pvals):
    """Holm step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def bonf_adjust(pvals, m=None):
    m = m or len(pvals)
    return [min(1.0, p * m) for p in pvals]


# ---------------------------------------------------------------- E7.1
def main_mult():
    names = list(DATASETS.keys())
    rows = []
    for hi, H in enumerate(HAZARDS):
        er = [ER_ORIG[n][hi] for n in names]
        for model, mi in [('iTransformer', 0), ('DLinear', 1)]:
            y = [BEST_SL[n][mi] for n in names]
            rho, p_exact = spearman_exact_p(er, y)
            _, p_t = spearman(er, y)  # t-approx, as in the published table
            rows.append(dict(H=H, model=model, rho=rho, p_exact=p_exact, p_tapprox=p_t))

    p_all = [r['p_exact'] for r in rows]
    holm10 = holm_adjust(p_all)
    bonf10 = bonf_adjust(p_all)
    for r, h10, b10 in zip(rows, holm10, bonf10):
        r['p_holm_m10'] = h10
        r['p_bonf_m10'] = b10
    for model in ['iTransformer', 'DLinear']:
        idxs = [i for i, r in enumerate(rows) if r['model'] == model]
        holm5 = holm_adjust([rows[i]['p_exact'] for i in idxs])
        bonf5 = bonf_adjust([rows[i]['p_exact'] for i in idxs])
        for i, h5, b5 in zip(idxs, holm5, bonf5):
            rows[i]['p_holm_m5'] = h5
            rows[i]['p_bonf_m5'] = b5

    lines = ['# E7.1 multiple-comparison correction of the exp4 BOCPD hazard scan', '',
             'Exact permutation p (n=8, 8! enumeration, ties with multiplicity);',
             't-approx p as printed in logs/mechanism/exp4/bocpd_summary.md for continuity.', '',
             '| H | model | rho | p_exact | p_tapprox | Holm (m=10) | Bonf (m=10) | Holm (m=5) | Bonf (m=5) |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['H']:g} | {r['model']} | {r['rho']:.3f} | {r['p_exact']:.4f} | "
                     f"{r['p_tapprox']:.3f} | {r['p_holm_m10']:.4f} | {r['p_bonf_m10']:.4f} | "
                     f"{r['p_holm_m5']:.4f} | {r['p_bonf_m5']:.4f} |")
    lines.append('')
    key = next(r for r in rows if r['model'] == 'iTransformer' and abs(r['H'] - 0.02) < 1e-9)
    lines.append(f"headline cell (iTransformer, H=1/50): rho={key['rho']:.3f}, "
                 f"p_exact={key['p_exact']:.4f} (t-approx {key['p_tapprox']:.3f}), "
                 f"Holm m=10 -> {key['p_holm_m10']:.4f}, Bonf m=10 -> {key['p_bonf_m10']:.4f}, "
                 f"Holm m=5 -> {key['p_holm_m5']:.4f}, Bonf m=5 -> {key['p_bonf_m5']:.4f}")
    lines.append(f"significant at 0.05 after m=10 Holm: {key['p_holm_m10'] < 0.05}; "
                 f"after m=5 Holm: {key['p_holm_m5'] < 0.05}; uncorrected exact: {key['p_exact'] < 0.05}")
    out_md = os.path.join(OUTD, 'e7_1_multcorr.md')
    with open(out_md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(OUTD, 'e7_1_multcorr.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print('\n'.join(lines))


# ---------------------------------------------------------------- E7.2
def load_train_raw(name):
    """Identical to bocpd_ehs.load_train minus standardization (applied post-transform)."""
    path, take = DATASETS[name]
    df = pd.read_csv(os.path.join(ROOT, path))
    data = df.drop(columns=['date']).values.astype(np.float64)
    n = take if isinstance(take, int) else int(len(data) * take)
    data = data[:n]
    C = data.shape[1]
    idx = np.linspace(0, C - 1, min(24, C)).astype(int)
    return data[:, idx]


def deseasonalize(x, P, variant):
    if variant == 'phase':
        tmod = np.arange(len(x)) % P
        pm = np.stack([x[tmod == p].mean(0) for p in range(P)])
        y = x - pm[tmod]
    elif variant == 'diff':
        y = x[P:] - x[:-P]
    else:
        raise ValueError(variant)
    mu = y.mean(0, keepdims=True)
    sd = y.std(0, keepdims=True) + 1e-8
    return ((y - mu) / sd).astype(np.float64)


def main_deseas_worker(variant, dataset):
    t0 = time.time()
    P = PERIODS[dataset]
    x = deseasonalize(load_train_raw(dataset), P, variant)
    res = {}
    for H in HAZARDS:
        Er, Pcp = bocpd_runlength(x, H)
        res[f'{H:g}'] = dict(Er=float(Er[BURN_IN:].mean()), Pcp=float(Pcp[BURN_IN:].mean()))
        print(f'[{variant}/{dataset}] H={H:g} Er={res[f"{H:g}"]["Er"]:.1f} ({time.time()-t0:.0f}s)',
              flush=True)
    out = dict(dataset=dataset, variant=variant, period=P, T=int(x.shape[0]), res=res)
    path = os.path.join(OUTD, f'deseas_{variant}_{dataset}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[{variant}/{dataset}] saved {path} ({time.time()-t0:.0f}s)', flush=True)


def spearman_table(er_by_ds, label):
    names = list(DATASETS.keys())
    lines = []
    records = []
    for hi, H in enumerate(HAZARDS):
        er = [er_by_ds[n][hi] for n in names]
        rec = {'H': H}
        msg = f'- H={H:g} [{label}]: '
        for model, mi in [('iTransformer', 0), ('DLinear', 1)]:
            y = [BEST_SL[n][mi] for n in names]
            rho, p = spearman_exact_p(er, y)
            rec[model] = dict(rho=rho, p_exact=p)
            msg += f'{model} rho={rho:.3f} (p_exact~{p:.3f})  '
        lines.append(msg)
        records.append(rec)
    return lines, records


def main_deseas_merge():
    names = list(DATASETS.keys())
    variants = {}
    for variant in ['phase', 'diff']:
        er = {}
        ok = True
        for n in names:
            path = os.path.join(OUTD, f'deseas_{variant}_{n}.json')
            if not os.path.exists(path):
                print(f'missing {path}')
                ok = False
                continue
            with open(path) as f:
                d = json.load(f)
            er[n] = [d['res'][f'{H:g}']['Er'] for H in HAZARDS]
        if ok:
            variants[variant] = er

    lines = ['# E7.2 deseasonalized BOCPD re-run (E[r], train split, same protocol as exp4)', '',
             'periods: ETTh 24, ETTm 96, exchange 7, weather 144, electricity/traffic 24.',
             'phase = subtract per-phase means; diff = seasonal difference at the same lag.', '']
    out_json = {}
    header = '| dataset | ' + ' | '.join(f'H={h:g}' for h in HAZARDS) + ' | best sl iTr | best sl DLin |'
    sep = '|---|' + '---|' * (len(HAZARDS) + 2)
    for label, er in [('original', ER_ORIG)] + list(variants.items()):
        lines += [f'## {label}', '', header, sep]
        for n in names:
            lines.append(f"| {n} | " + ' | '.join(f'{v:.0f}' for v in er[n]) +
                         f' | {BEST_SL[n][0]} | {BEST_SL[n][1]} |')
        lines.append('')
        tl, rec = spearman_table(er, label)
        lines += tl + ['']
        out_json[label] = dict(Er=er, spearman=rec)
        # electricity/traffic focus
        for n in ['electricity', 'traffic']:
            ranks = [int(np.sum([er[m][hi] < er[n][hi] for m in names]) + 1) for hi in range(len(HAZARDS))]
            lines.append(f'  {n}: E[r] rank among 8 (1=shortest) per hazard = {ranks}, '
                         f'best sl = {BEST_SL[n][0]}/{BEST_SL[n][1]} (tied longest)')
        lines.append('')

    path_md = os.path.join(OUTD, 'e7_2_deseas.md')
    with open(path_md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(OUTD, 'e7_2_deseas.json'), 'w') as f:
        json.dump(out_json, f, indent=2, default=float)
    print('\n'.join(lines))


# ---------------------------------------------------------------- E7.3
def rolling_meandiff(x, w, stride=8):
    """Mean |mean(x[t-w:t]) - mean(x[t:t+w])| over t and channels."""
    T, C = x.shape
    cs = np.cumsum(x, axis=0)
    ts = np.arange(w, T - w, stride)
    m1 = (cs[ts] - cs[ts - w]) / w
    m2 = (cs[ts + w] - cs[ts]) / w
    return float(np.abs(m1 - m2).mean())


def rolling_kl(x, w, stride=8, vfloor=1e-3):
    """Symmetrized Gaussian KL between adjacent w-windows (per channel).
    Variance floored at vfloor (standardized units) so near-degenerate windows
    do not dominate; both mean and median over t are returned."""
    T, C = x.shape
    cs = np.cumsum(x, axis=0)
    cs2 = np.cumsum(x ** 2, axis=0)
    ts = np.arange(w, T - w, stride)
    mu1 = (cs[ts] - cs[ts - w]) / w
    mu2 = (cs[ts + w] - cs[ts]) / w
    v1 = np.maximum((cs2[ts] - cs2[ts - w]) / w - mu1 ** 2, vfloor)
    v2 = np.maximum((cs2[ts + w] - cs2[ts]) / w - mu2 ** 2, vfloor)
    kl12 = 0.5 * (np.log(v2 / v1) + (v1 + (mu1 - mu2) ** 2) / v2 - 1.0)
    kl21 = 0.5 * (np.log(v1 / v2) + (v2 + (mu1 - mu2) ** 2) / v1 - 1.0)
    skl = 0.5 * (kl12 + kl21)
    return float(skl.mean()), float(np.median(skl))


def drift_index(x, n_seg=20):
    """ehs_stats segment-mean-variance drift index (reference, not independent)."""
    out = []
    for j in range(x.shape[1]):
        segs = np.array_split(x[:, j], n_seg)
        means = np.array([s.mean() for s in segs if len(s)])
        out.append(means.var() / (x[:, j].var() + 1e-12))
    return float(np.mean(out))


def main_drift():
    names = list(DATASETS.keys())
    stats = {}
    for n in names:
        x = load_train_raw(n)
        mu = x.mean(0, keepdims=True)
        sd = x.std(0, keepdims=True) + 1e-8
        x = (x - mu) / sd
        kl96_m, kl96_med = rolling_kl(x, 96)
        kl336_m, kl336_med = rolling_kl(x, 336)
        stats[n] = dict(
            meandiff_w96=rolling_meandiff(x, 96),
            meandiff_w336=rolling_meandiff(x, 336),
            kl_w96=kl96_m,
            kl_w96_med=kl96_med,
            kl_w336=kl336_m,
            kl_w336_med=kl336_med,
            drift_index_20seg=drift_index(x),
        )
        print(f'[drift] {n}: {stats[n]}', flush=True)

    lines = ['# E7.3 BOCPD-free drift statistics vs measured best seq_len', '',
             'train split, <=24 linspace channels, standardized (same data as exp4).',
             'meandiff_w: mean |mean diff| of adjacent w-windows; kl_w: symmetrized',
             'Gaussian KL of adjacent w-windows (window variance floored at 1e-3,',
             'standardized units; mean and median over t). All changepoint-model-free.', '',
             '| dataset | meandiff w96 | meandiff w336 | KL w96 mean/med | KL w336 mean/med | drift_idx(20seg) | best iTr | best DLin |',
             '|---|---|---|---|---|---|---|---|']
    for n in names:
        s = stats[n]
        lines.append(f"| {n} | {s['meandiff_w96']:.4f} | {s['meandiff_w336']:.4f} | "
                     f"{s['kl_w96']:.3f}/{s['kl_w96_med']:.3f} | {s['kl_w336']:.3f}/{s['kl_w336_med']:.3f} | "
                     f"{s['drift_index_20seg']:.4f} | {BEST_SL[n][0]} | {BEST_SL[n][1]} |")
    lines.append('')
    out_json = {'stats': stats, 'spearman': {}}
    rows = []
    for key in ['meandiff_w96', 'meandiff_w336', 'kl_w96', 'kl_w96_med',
                'kl_w336', 'kl_w336_med', 'drift_index_20seg']:
        v = [stats[n][key] for n in names]
        for model, mi in [('iTransformer', 0), ('DLinear', 1)]:
            y = [BEST_SL[n][mi] for n in names]
            rho, p = spearman_exact_p(v, y)
            rows.append(dict(stat=key, model=model, rho=rho, p_exact=p))
            out_json['spearman'][f'{key}/{model}'] = dict(rho=rho, p_exact=p)
    holm14 = holm_adjust([r['p_exact'] for r in rows])
    for r, h14 in zip(rows, holm14):
        r['p_holm_m14'] = h14
    for model in ['iTransformer', 'DLinear']:
        idxs = [i for i, r in enumerate(rows) if r['model'] == model]
        h7 = holm_adjust([rows[i]['p_exact'] for i in idxs])
        for i, hh in zip(idxs, h7):
            rows[i]['p_holm_m7'] = hh
    for r in rows:
        lines.append(f"- {r['stat']} vs best sl [{r['model']}]: rho={r['rho']:.3f} "
                     f"(p_exact~{r['p_exact']:.3f}, Holm m=7 {r['p_holm_m7']:.3f}, "
                     f"Holm m=14 {r['p_holm_m14']:.3f})")
        out_json['spearman'][f"{r['stat']}/{r['model']}"] = dict(
            rho=r['rho'], p_exact=r['p_exact'], p_holm_m7=r['p_holm_m7'], p_holm_m14=r['p_holm_m14'])
    lines.append('')
    lines.append('Expectation under the EHS drift story: higher drift -> shorter best '
                 'lookback -> NEGATIVE rho supports the story independent of BOCPD.')
    with open(os.path.join(OUTD, 'e7_3_drift.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(OUTD, 'e7_3_drift.json'), 'w') as f:
        json.dump(out_json, f, indent=2)
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['mult', 'deseas-worker', 'deseas-merge', 'drift'])
    ap.add_argument('--variant', choices=['phase', 'diff'])
    ap.add_argument('--dataset', choices=list(DATASETS.keys()))
    args = ap.parse_args()
    if args.mode == 'mult':
        main_mult()
    elif args.mode == 'deseas-worker':
        main_deseas_worker(args.variant, args.dataset)
    elif args.mode == 'deseas-merge':
        main_deseas_merge()
    elif args.mode == 'drift':
        main_drift()
