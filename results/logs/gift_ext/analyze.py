#!/usr/bin/env python
"""EHS x GIFT-Eval correlation analysis: do the two axes hold at GIFT scale?

Targets per task (from scan/*.npz + scan_summary.csv):
  best_L / log2_best_L     argmin CRPS over L
  ehs1pct / log2_ehs1pct   smallest L within 1% of min CRPS (EHS 定义: 收益<eps 的最小 L)
  benefit / rel_benefit    CRPS@96 - CRPS@2880, absolute and relative
  (same three families for nmse = seasonal-naive-normalized MSE)
  benefit_boot_p           bootstrap P(benefit > 0) over items (1000 resamples)

Stats (stats.csv, analysis/ehs_stats.py 口径, train segment only):
  drift (axis 1), calendar_acf_diff + long_acf_diff (axis 2), periodicity_dt, long_acf.

Outputs: logs/gift_ext/analysis.csv, logs/gift_ext/correlations.csv,
         logs/gift_ext/fig_two_axes.png, printed verdict.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LS = [96, 336, 720, 1440, 2880]
STATS = ['drift', 'calendar_acf_diff', 'long_acf_diff', 'periodicity_dt', 'long_acf']
# previous 8-dataset x 2-model reference (logs/ehs_final, benefit = mse96-mse2880)
PREV = {'drift': (-0.813, 0.001), 'calendar_acf_diff': (0.694, 0.004),
        'long_acf_diff': (0.630, 0.012)}
QLEVELS = np.arange(0.1, 1.0, 0.1)
SCAN_DIR = 'logs/gift_ext/scan'
EPS = 0.01


def per_item_crps(quants, tgts):
    """quants [nL, M, 9, P] -> per-item CRPS [nL, M] (NaN-target masked)."""
    y = tgts.astype(np.float64)[None]            # [1, M, P]
    q = quants.astype(np.float64)                # [nL, M, 9, P]
    vt = np.isfinite(y) & np.isfinite(q).all(axis=2)
    pin = np.abs(y[:, :, None, :] - q) * np.where(y[:, :, None, :] >= q,
                                                  QLEVELS[None, None, :, None],
                                                  1 - QLEVELS[None, None, :, None])
    ql = np.where(vt[:, :, None, :], pin, 0.0).sum(axis=(2, 3))      # [nL, M]
    denom = np.abs(np.where(vt, y, 0.0)).sum(axis=2)                  # [1, M]
    n_vt = vt.sum(axis=2)
    ok = (n_vt > 0) & (denom > 1e-8)
    return np.where(ok, 2 * ql / np.maximum(denom, 1e-8), np.nan)     # [nL, M]


def per_item_nmse(quants, tgts, snaives):
    y = tgts.astype(np.float64)[None]
    point = np.nanmean(quants.astype(np.float64), axis=2)             # [nL, M, P]
    vt = np.isfinite(y) & np.isfinite(point)
    n_vt = vt.sum(axis=2)
    se = (np.where(vt, (point - y) ** 2, 0.0)).sum(axis=2) / np.maximum(n_vt, 1)
    sn = (np.where(vt, (snaives.astype(np.float64)[None] - y) ** 2, 0.0)).sum(axis=2) / np.maximum(n_vt, 1)
    return np.where((n_vt > 0) & (sn > 1e-12), se / np.maximum(sn, 1e-12), np.nan)


def task_targets(pi):
    """pi: [nL, M] per-item metric. Returns dict of task-level targets."""
    per_L = np.nanmean(pi, axis=1)                       # [nL]
    best = LS[int(np.nanargmin(per_L))]
    thr = np.nanmin(per_L) * (1 + EPS)
    ehs = LS[int(np.argmax(per_L <= thr))]               # smallest L within EPS of min
    ben96 = per_L[0] - per_L[-1]
    out = {'best_L': best, 'log2_best_L': float(np.log2(best)),
           'ehs1pct': ehs, 'log2_ehs1pct': float(np.log2(ehs)),
           'benefit': float(ben96),
           'rel_benefit': float(ben96 / per_L[0])}
    # bootstrap P(benefit>0): resample per-item ABSOLUTE differences
    # (per-item ratios explode on near-zero denominators; the aggregate benefit
    # is a ratio of means, so bootstrap the difference, same estimand)
    rng = np.random.default_rng(0)
    d = pi[0] - pi[-1]
    d = d[np.isfinite(d)]
    if len(d) >= 5:
        boots = rng.choice(d, size=(1000, len(d)), replace=True).mean(axis=1)
        out['benefit_boot_p_gt0'] = float((boots > 0).mean())
    else:
        out['benefit_boot_p_gt0'] = np.nan
    return out


def build():
    stats = pd.read_csv('logs/gift_ext/stats.csv')
    meta = json.load(open('logs/gift_ext/tasks.json'))
    rows = []
    for task in meta:
        z = np.load(os.path.join(SCAN_DIR, f'{task}.npz'), allow_pickle=True)
        quants, tgts, snaives = z['quants'], z['target'], z['snaive']
        nS, W, P = tgts.shape
        q_flat = quants.reshape(len(LS), nS * W, 9, P)
        t_flat = tgts.reshape(nS * W, P)
        s_flat = snaives.reshape(nS * W, P)
        row = {'task': task}
        for suf, pi in [('crps', per_item_crps(q_flat, t_flat)),
                        ('nmse', per_item_nmse(q_flat, t_flat, s_flat))]:
            for k, v in task_targets(pi).items():
                row[f'{k}_{suf}'] = v
        s = stats[stats.task == task].iloc[0]
        for k in STATS:
            row[k] = s[f'{k}_mean']
        for k in ['domain', 'freq', 'pred_len', 'windows', 'n_series', 'context_capped']:
            row[k] = s[k]
        rows.append(row)
    return pd.DataFrame(rows)


def corr_block(df, label):
    targets = ['rel_benefit_crps', 'benefit_crps', 'log2_best_L_crps', 'log2_ehs1pct_crps',
               'rel_benefit_nmse', 'log2_best_L_nmse', 'log2_ehs1pct_nmse']
    out = []
    print(f'\n== Spearman correlations ({label}, n={len(df)}) ==')
    for t in targets:
        for k in STATS:
            d = df[[k, t]].dropna()
            if len(d) < 5 or d[k].nunique() < 3:
                continue
            r, p = spearmanr(d[k], d[t])
            out.append({'stat': k, 'target': t, 'n': len(d), 'spearman': float(r),
                        'p': float(p), 'subset': label})
            sig = '*' if p < 0.05 else ' '
            print(f'{k:18} vs {t:22} rho={r:+.3f} p={p:.4f} {sig}')
    return out


def fig(df):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    cmap = {96: '#d62728', 336: '#ff7f0e', 720: '#2ca02c', 1440: '#1f77b4', 2880: '#9467bd'}
    for L in LS:
        sub = df[df.ehs1pct_crps == L]
        if len(sub) == 0:
            continue
        ax.scatter(sub.drift, sub.calendar_acf_diff, s=30 + 130 * sub.rel_benefit_crps.clip(0),
                   c=cmap[L], label=f'EHS={L}', alpha=0.85, edgecolors='k', linewidths=0.5)
    for _, r in df.iterrows():
        ax.annotate(r.task.replace('_with_missing', '').replace('hierarchical', 'hier'),
                    (r.drift, r.calendar_acf_diff), fontsize=5.5, alpha=0.75,
                    xytext=(3, 3), textcoords='offset points')
    ax.axhline(0, c='gray', lw=0.5); ax.axvline(0, c='gray', lw=0.5)
    ax.set_xlabel('axis 1: drift index (segment-mean var / total var)')
    ax.set_ylabel('axis 2: calendar ACF (detrended+differenced)')
    ax.set_title('Two-axis map, 25 GIFT-Eval tasks\n(color=EHS by CRPS @1% tol, size=rel. long-context benefit)')
    ax.legend(fontsize=7)
    ax = axes[1]
    order = df.sort_values('z_gap')
    colors = ['#d62728' if g > 0 else '#2ca02c' for g in order.z_gap]
    ax.barh(order.task.str.replace('_with_missing', ''), order.z_gap, color=colors)
    ax.axvline(0, c='k', lw=0.6)
    ax.set_xlabel('z(drift) - z(calendar_acf)   (>0 drift-dominated)')
    ax.set_title('Dominant axis per task')
    ax.tick_params(axis='y', labelsize=6)
    fig.tight_layout()
    fig.savefig('logs/gift_ext/fig_two_axes.png', dpi=160)
    print('wrote logs/gift_ext/fig_two_axes.png')


def main():
    df = build()
    zd = (df.drift - df.drift.mean()) / df.drift.std()
    zc = (df.calendar_acf_diff - df.calendar_acf_diff.mean()) / df.calendar_acf_diff.std()
    df['z_drift'], df['z_cal'] = zd, zc
    df['z_gap'] = zd - zc
    df['dominant'] = np.where(df.z_gap > 0, 'drift', 'period')
    df.to_csv('logs/gift_ext/analysis.csv', index=False)

    print('== Per-task summary ==')
    cols = ['task', 'domain', 'freq', 'n_series', 'best_L_crps', 'ehs1pct_crps',
            'rel_benefit_crps', 'benefit_boot_p_gt0_crps', 'drift', 'calendar_acf_diff',
            'long_acf_diff', 'dominant']
    print(df[cols].round(3).to_string(index=False))

    all_rows = corr_block(df, 'all25')
    all_rows += corr_block(df[~df.context_capped], 'no_capped24')
    all_rows += corr_block(df[df.n_series >= 20], 'n_series>=20')
    pd.DataFrame(all_rows).to_csv('logs/gift_ext/correlations.csv', index=False)

    print('\n== Dominance ==')
    print(df.groupby('dominant').agg(n=('task', 'count'), tasks=('task', lambda s: ', '.join(s))).to_string())
    print('\nbenefit>0 (rel CRPS):', int((df.rel_benefit_crps > 0).sum()), '/', len(df),
          '; bootstrap p>0.95:', int((df.benefit_boot_p_gt0_crps > 0.95).sum()),
          '; p<0.05:', int((df.benefit_boot_p_gt0_crps < 0.05).sum()))
    fig(df)

    print('\n== Verdict vs previous 8-dataset x 2-model result ==')
    for k, (pr, pp) in PREV.items():
        r, p = spearmanr(df[k], df['rel_benefit_crps'])
        r2, p2 = spearmanr(df[k], df['log2_ehs1pct_crps'])
        print(f'{k:18} prev rho={pr:+.3f}(p={pp:.3f}) -> gift25: '
              f'vs rel_benefit rho={r:+.3f}(p={p:.3f}) [{"KEPT" if np.sign(r)==np.sign(pr) else "FLIPPED"}], '
              f'vs log2 EHS rho={r2:+.3f}(p={p2:.3f})')


if __name__ == '__main__':
    main()
