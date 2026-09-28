#!/usr/bin/env python
"""EHS predictability v3: extended to 14 datasets (8 old + 6 new).

Same statistics and protocol as analysis/ehs_predict_v2.py, but:
- DATASETS adds solar (10-min), PEMS04/PEMS08 (5-min), ercot/pedestrian/bikes (hourly).
- Targets parsed from BOTH logs/ehs_v2/SUMMARY.md (old 8) and
  logs/ehs_v2_ext/SUMMARY.md (new 6). 28 samples = 14 datasets x 2 models.
- Outputs to logs/ehs_v2_ext/.

Usage: .venv/bin/python analysis/ehs_predict_v3.py
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ehs_stats import (acf_at, detrend, drift_index, long_acf,
                                periodicity_detrended)

DATASETS = {
    'ETTh1':        ('dataset/ETT-small/ETTh1.csv', [24, 168]),
    'ETTh2':        ('dataset/ETT-small/ETTh2.csv', [24, 168]),
    'ETTm1':        ('dataset/ETT-small/ETTm1.csv', [96, 672]),
    'ETTm2':        ('dataset/ETT-small/ETTm2.csv', [96, 672]),
    'exchange_rate': ('dataset/exchange_rate/exchange_rate.csv', [5, 30]),
    'weather':      ('dataset/weather/weather.csv', [144, 1008]),
    'electricity':  ('dataset/electricity/electricity.csv', [24, 168]),
    'traffic':      ('dataset/traffic/traffic.csv', [24, 168]),
    'solar':        ('dataset/solar/solar.csv', [144, 1008]),
    'PEMS04':       ('dataset/PEMS04/PEMS04.csv', [288, 2016]),
    'PEMS08':       ('dataset/PEMS08/PEMS08.csv', [288, 2016]),
    'ercot':        ('dataset/ercot/ercot.csv', [24, 168]),
    'pedestrian':   ('dataset/pedestrian/pedestrian.csv', [24, 168]),
    'bikes':        ('dataset/bikes/bikes.csv', [24, 168]),
}
SEQ_LENS = [96, 336, 720, 1440, 2880]
STAT_NAMES = ['periodicity_dt', 'drift', 'long_acf', 'calendar_acf_diff', 'long_acf_diff']
OUT_DIR = 'logs/ehs_v2_ext'

# Train-split row counts (TSLib borders: ETT fixed calendar splits, others 70/10/20;
# verified against tab:nmin and the per-run "train N" log lines). Used by
# --convention train: statistics are computed on the train split only, so no
# validation/test information enters the features (deployment-realistic).
TRAIN_ROWS = {
    'ETTh1': 8640, 'ETTh2': 8640, 'ETTm1': 34560, 'ETTm2': 34560,
    'exchange_rate': 5311, 'weather': 36887, 'electricity': 18412, 'traffic': 12280,
    'solar': 36792, 'PEMS04': 11894, 'PEMS08': 12499,
    'ercot': 108410, 'pedestrian': 59031, 'bikes': 36825,
}


def calendar_acf_diff(x, periods):
    xd = np.diff(detrend(x - x.mean()))
    return float(np.nanmean([acf_at(xd, p) for p in periods]))


def long_acf_diff(x, lag_frac=0.25):
    xd = np.diff(detrend(x - x.mean()))
    return acf_at(xd, max(1, int(len(xd) * lag_frac)))


def channel_stats(x, periods):
    return {
        'periodicity_dt': periodicity_detrended(x),
        'drift': drift_index(x),
        'long_acf': long_acf(x),
        'calendar_acf_diff': calendar_acf_diff(x, periods),
        'long_acf_diff': long_acf_diff(x),
    }


def dataset_stats(path, periods, n_channels=50, max_len=12000, seed=0, train_rows=None):
    df = pd.read_csv(path)
    X = df.drop(columns=[c for c in df.columns if c.lower() == 'date']).values.astype(float)
    if train_rows is not None:
        X = X[:train_rows]
    X = X[:max_len]
    rng = np.random.default_rng(seed)
    ch = rng.choice(X.shape[1], size=min(n_channels, X.shape[1]), replace=False)
    per_ch = [channel_stats(X[:, j], periods) for j in ch]
    out = {}
    for k in STAT_NAMES:
        v = np.array([s[k] for s in per_ch], dtype=float)
        out[k] = (float(np.nanmean(v)), float(np.nanstd(v)), int(np.sum(~np.isnan(v))))
    return out


def parse_summary_mse(paths):
    """Parse '### Model: <m> (MSE)' tables from one or more SUMMARY.md files."""
    mse, best = {'iTransformer': {}, 'DLinear': {}}, {'iTransformer': {}, 'DLinear': {}}
    for summary_path in paths:
        txt = open(summary_path).read()
        for model in ['iTransformer', 'DLinear']:
            sec = re.search(rf'### Model: {model} \(MSE\)\n(.*?)(?:\n###|\Z)', txt, re.S)
            if sec is None:
                continue
            for line in sec.group(1).splitlines():
                m = re.match(r'\| ([\w ]*?) \| (.*) \| (\d+) \|$', line)
                if not m or m.group(1) == 'dataset':
                    continue
                ds, cells, b = m.group(1).strip(), m.group(2), int(m.group(3))
                vals = {}
                for sl, cell in zip(SEQ_LENS, cells.split(' | ')):
                    cell = cell.replace('*', '').strip()
                    if cell == '-':
                        continue
                    vals[sl] = float(re.match(r'([\d.]+)', cell).group(1))
                mse[model][ds] = vals
                best[model][ds] = b
    return mse, best


def ridge_fit(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    ymu = y.mean()
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs.T @ (y - ymu))
    return (w, mu, sd, ymu)


def ridge_pred(model, X):
    w, mu, sd, ymu = model
    return (X - mu) / sd @ w + ymu


def main():
    convention = 'full'
    if '--convention' in sys.argv:
        convention = sys.argv[sys.argv.index('--convention') + 1]
    assert convention in ('full', 'train')
    suffix = '' if convention == 'full' else '_train'
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []

    def out(s=''):
        print(s)
        lines.append(s)

    out(f'== Dataset statistics (mean over <=50 channels, max_len=12000, convention={convention}) ==')
    stats = {}
    for ds, (path, periods) in DATASETS.items():
        tr = TRAIN_ROWS[ds] if convention == 'train' else None
        stats[ds] = dataset_stats(path, periods, train_rows=tr)
        cell = '  '.join(f'{k}={stats[ds][k][0]:+.4f}' for k in STAT_NAMES)
        out(f'{ds:15} {cell}')

    mse, best = parse_summary_mse(['logs/ehs_v2/SUMMARY.md', 'logs/ehs_v2_ext/SUMMARY.md'])
    rows = []
    for model in ['iTransformer', 'DLinear']:
        for ds in DATASETS:
            v = mse[model][ds]
            benefit = v[96] - v[2880]
            rows.append({'dataset': ds, 'model': model,
                         'best_sl': best[model][ds],
                         'log2_best_sl': float(np.log2(best[model][ds])),
                         'benefit': benefit,
                         **{k: stats[ds][k][0] for k in STAT_NAMES}})
            out(f'target {model:12} {ds:15} best_sl={best[model][ds]:5} benefit={benefit:+.4f}')

    df = pd.DataFrame(rows)
    df.to_json(os.path.join(OUT_DIR, f'taskB_samples_v3{suffix}.json'), orient='records', indent=1)

    n = len(df)
    out(f'\n== Correlations with targets (n={n} = {n // 2} datasets x 2 models) ==')
    corr = {}
    for k in STAT_NAMES:
        x = df[k].values.astype(float)
        sb, pb = spearmanr(x, df['benefit']); rb, _ = pearsonr(x, df['benefit'])
        sl_, pl = spearmanr(x, df['log2_best_sl']); rl, _ = pearsonr(x, df['log2_best_sl'])
        corr[k] = {'spearman_benefit': (float(sb), float(pb)),
                   'pearson_benefit': (float(rb), float(pb)),
                   'spearman_log2sl': (float(sl_), float(pl)),
                   'pearson_log2sl': (float(rl), float(pl))}
        out(f'{k:18} vs benefit: spearman={sb:+.3f}(p={pb:.3f}) pearson={rb:+.3f} | '
            f'vs log2(best_sl): spearman={sl_:+.3f}(p={pl:.3f}) pearson={rl:+.3f}')

    # dataset-level view (n=14): average the two model samples per dataset; these
    # are the statistically independent units (features are identical within a dataset)
    ds_df = df.groupby('dataset', sort=False).agg(
        {**{k: 'first' for k in STAT_NAMES}, 'benefit': 'mean', 'log2_best_sl': 'mean'})
    n14 = len(ds_df)
    out(f'\n== Dataset-level correlations (n={n14}, model-averaged targets) ==')
    corr_ds = {}
    for k in STAT_NAMES:
        x = ds_df[k].values.astype(float)
        sb, pb = spearmanr(x, ds_df['benefit'])
        sl_, pl = spearmanr(x, ds_df['log2_best_sl'])
        corr_ds[k] = {'spearman_benefit': (float(sb), float(pb)),
                      'spearman_log2sl': (float(sl_), float(pl))}
        out(f'{k:18} vs benefit: spearman={sb:+.3f}(p={pb:.4f}) | '
            f'vs log2(best_sl): spearman={sl_:+.3f}(p={pl:.4f})')

    out('\n== Leave-one-dataset-out CV (ridge, features = top-k |spearman| inside fold) ==')
    tiers = np.array(SEQ_LENS, dtype=float)
    loo = {}
    for k_feat in (2, 3):
        preds, trues, hit, correct_dir = [], [], 0, 0
        detail = []
        for ds in DATASETS:
            tr = df[df.dataset != ds]
            te = df[df.dataset == ds]
            rank = sorted(STAT_NAMES,
                          key=lambda k: abs(spearmanr(tr[k], tr['log2_best_sl'])[0]),
                          reverse=True)[:k_feat]
            m = ridge_fit(tr[rank].values.astype(float), tr['log2_best_sl'].values)
            for _, r in te.iterrows():
                p = float(ridge_pred(m, r[rank].values.astype(float)))
                preds.append(p)
                trues.append(r['log2_best_sl'])
                p_tier = int(tiers[np.argmin(np.abs(tiers - 2 ** np.clip(p, np.log2(96), np.log2(2880))))])
                hit += int(p_tier == r['best_sl'])
                correct_dir += int(np.sign(p - df['log2_best_sl'].median()) ==
                                   np.sign(r['log2_best_sl'] - df['log2_best_sl'].median()))
                detail.append({'dataset': ds, 'model': r['model'], 'true_sl': int(r['best_sl']),
                               'pred_log2': p, 'pred_tier': p_tier, 'features': rank})
        sp, pp = spearmanr(preds, trues)
        base_hit = sum(int(int(tiers[np.argmin(np.abs(tiers - 2 ** df['log2_best_sl'].median()))]) == t)
                       for t in df['best_sl'])
        loo[f'k{k_feat}'] = {'spearman': (float(sp), float(pp)), 'tier_hit': hit,
                             'n': len(trues), 'dir_acc': correct_dir, 'detail': detail}
        out(f'k={k_feat}: LOO spearman(pred,true log2 sl)={sp:+.3f} (p={pp:.3f}), '
            f'exact tier hit {hit}/{len(trues)} (baseline const-median {base_hit}/{len(trues)}), '
            f'direction acc {correct_dir}/{len(trues)}')
        for d in detail:
            out(f"    {d['dataset']:15} {d['model']:12} true={d['true_sl']:5} "
                f"pred_tier={d['pred_tier']:5} (pred_log2={d['pred_log2']:+.2f}) feats={d['features']}")

    verdict = {
        'corr': corr,
        'corr_dataset_level': corr_ds,
        'loo': {k: {kk: vv for kk, vv in v.items() if kk != 'detail'} for k, v in loo.items()},
    }
    with open(os.path.join(OUT_DIR, f'taskB_results_v3{suffix}.json'), 'w') as f:
        json.dump(verdict, f, indent=1)
    with open(os.path.join(OUT_DIR, f'taskB_predict_v3{suffix}.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nwrote {OUT_DIR}/taskB_predict_v3{suffix}.txt, taskB_results_v3{suffix}.json, '
          f'taskB_samples_v3{suffix}.json')


if __name__ == '__main__':
    main()
