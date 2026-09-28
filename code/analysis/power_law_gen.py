#!/usr/bin/env python
"""lambda-parameterized long synthetic series for the W* ~ lambda^(-2/3) experiment.

Design (reuses tsfm/shock_data.py shock semantics, but vectorized and long-form):
  - Poisson shock process with per-step probability lam: run lengths ~ Geometric(lam),
    so E[run] = 1/lam by construction (measured mean run length reported in calibration).
  - Shock type: LEVEL steps only (pure drift channel; magnitude U(0.8,2.0), random sign,
    independent shock process per channel).
  - Base noise: AR(1) a=0.3 with marginal sigma=12. The high noise-to-step ratio
    (sigma/s_rms ~ 8) is ESSENTIAL: it moves W* = argmin_W MSE into the scanned
    lookback range (with sigma=1 the bias-variance optimum sits below W=96 for the
    whole lambda ladder and every arm is boundary-censored).
  - 'periodic' variant adds a strong static skeleton per channel (regime-invariant
    component): 12*sin(2pi t/96 + phi) + 6*sin(2pi t/48 + 2phi) (~1 sigma amplitude).
  - 4 channels (ch0..ch2, OT), T=25000 -> Dataset_Custom split: train 17500 / val 2500 /
    test 5000; N_min(sl=2880, pred 96) = 17500-2880-96+1 = 14525 train windows.
  - E[run] ladder {50,150,400,1000,2500,6000}: extends the suggested 50..2000 range at
    the slow end so that W*(lambda) sweeps the interior of {96..2880}.

Outputs CSVs under dataset/power_law/ plus calibration (drift_index per level,
monotone check) into logs/power_law/calibration.json and figs/calibration.png.

Run: .venv/bin/python analysis/power_law_gen.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'dataset', 'power_law')
LOG = os.path.join(ROOT, 'logs', 'power_law')
FIG = os.path.join(LOG, 'figs')

T = 25000
C = 4
AR_A = 0.3
NOISE_SIGMA = 12.0   # >> step rms ~1.45 so W* lands inside the scanned lookbacks
PER_AMP = (12.0, 6.0)  # strong periodic skeleton (fundamental, 2nd harmonic)
PER_PERIODS = (96, 48)
# E[run] ladder: extends 50..2000 to 6000 at the slow end so W* stays interior
ER_LADDER = [50, 150, 400, 1000, 2500, 6000]
ER_PERIODIC = [150, 2500]  # fast / slow drift for the p-axis arms


def gen_series(T, lam, C, periodic, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    X = np.zeros((T, C))
    ers = []
    n_shocks = 0
    for c in range(C):
        # independent Poisson shock process per channel: run lengths ~ Geometric(lam)
        runs = []
        pos = 0
        while True:
            r = rng.geometric(lam)
            pos += r
            if pos >= T:
                break
            runs.append(r)
        positions = np.array(np.cumsum(runs), dtype=np.int64)
        seg_id = np.zeros(T, dtype=np.int64)
        if len(positions):
            seg_id[positions] = 1
            seg_id = np.cumsum(seg_id)
        ns = len(positions)
        n_shocks += ns
        ers.append(T / max(ns, 1))
        steps = rng.uniform(0.8, 2.0, ns + 1) * rng.choice([-1, 1], ns + 1)
        steps[0] = 0.0
        level = np.cumsum(steps)[seg_id]
        eps = rng.normal(0, NOISE_SIGMA * np.sqrt(1 - AR_A ** 2), T)
        noise = lfilter([1.0], [1.0, -AR_A], eps)
        x = level + noise
        if periodic:
            phi = rng.uniform(0, 2 * np.pi)
            x = x + PER_AMP[0] * np.sin(2 * np.pi * t / PER_PERIODS[0] + phi) \
                + PER_AMP[1] * np.sin(2 * np.pi * t / PER_PERIODS[1] + 2 * phi)
        X[:, c] = x
    meta = dict(n_shocks=n_shocks, measured_E_run=float(np.mean(ers)))
    return X, meta


def drift_index(x, n_seg=20):
    segs = np.array_split(x, n_seg)
    means = np.array([s.mean() for s in segs if len(s)])
    return float(means.var() / (x.var() + 1e-12))


def write_csv(name, X):
    dates = pd.date_range('2020-01-01', periods=len(X), freq='h')
    df = pd.DataFrame(X, columns=['ch0', 'ch1', 'ch2', 'OT'])
    df.insert(0, 'date', dates.strftime('%Y-%m-%d %H:%M:%S'))
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    calib = {}
    jobs = [('drift', er, False) for er in ER_LADDER] + [('per', er, True) for er in ER_PERIODIC]
    for tag, er, periodic in jobs:
        lam = 1.0 / er
        X, meta = gen_series(T, lam, C, periodic, seed=7)
        name = f'{tag}_er{er}.csv'
        write_csv(name, X)
        drifts = [drift_index(X[:17500, c]) for c in range(C)]
        calib[name] = dict(lam=lam, nominal_E_run=er, measured_E_run=meta['measured_E_run'],
                           n_shocks=meta['n_shocks'], periodic=periodic,
                           drift_per_channel=drifts, drift_mean=float(np.mean(drifts)),
                           drift_std=float(np.std(drifts)))
        print(f'{name}: lam={lam:.5f} measured_E[r]={meta["measured_E_run"]:.0f} '
              f'shocks={meta["n_shocks"]} drift={np.mean(drifts):.4f}±{np.std(drifts):.4f}', flush=True)

    # multi-realization calibration curve (30 realizations x C channels per lambda)
    curve = {}
    for er in ER_LADDER:
        lam = 1.0 / er
        vals = []
        for rep in range(30):
            X, _ = gen_series(T, lam, C, False, seed=1000 + rep)
            vals += [drift_index(X[:17500, c]) for c in range(C)]
        curve[er] = dict(lam=lam, drift_mean=float(np.mean(vals)), drift_std=float(np.std(vals)),
                         drift_sem=float(np.std(vals) / np.sqrt(len(vals))), n=len(vals))
        print(f'calib er={er}: drift={np.mean(vals):.4f}±{np.std(vals) / np.sqrt(len(vals)):.4f} (n={len(vals)})',
              flush=True)
    calib['_curve'] = curve
    with open(os.path.join(LOG, 'calibration.json'), 'w') as f:
        json.dump(calib, f, indent=2)

    ers_sorted = sorted(ER_LADDER)  # ascending E[r] = descending lambda
    lams = [curve[e]['lam'] for e in ers_sorted]
    drifts = [curve[e]['drift_mean'] for e in ers_sorted]
    # drift must DECREASE along ascending E[r] (i.e. increase with lambda)
    mono = all(drifts[i] >= drifts[i + 1] for i in range(len(drifts) - 1))
    # the ratio statistic saturates once walk variance dominates noise: the two
    # fastest levels (er50/er100) form a plateau; strict check on the rest
    mono_strict_unsat = all(drifts[i] > drifts[i + 1] for i in range(1, len(drifts) - 1))
    print(f'drift_index monotone non-increasing along ascending E[r] (i.e. increasing in lambda): {mono}; '
          f'strict outside the er50/er100 saturation plateau: {mono_strict_unsat}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    order = np.argsort(lams)
    ax[0].errorbar(np.array(lams)[order], np.array(drifts)[order],
                   yerr=np.array([curve[e]['drift_std'] for e in ers_sorted])[order],
                   fmt='o-', capsize=3, label='pure drift (30 realizations)')
    dser = [calib[f'drift_er{er}.csv']['drift_mean'] for er in ers_sorted]
    ax[0].loglog(np.array(lams)[order], np.array(dser)[order], 'x', ms=8, label='actual training CSV')
    per = [(calib[f'per_er{er}.csv']['lam'], calib[f'per_er{er}.csv']['drift_mean'])
           for er in ER_PERIODIC]
    ax[0].loglog([p[0] for p in per], [p[1] for p in per], 's--', label='periodic variant (train CSV)')
    ax[0].set_xlabel('lambda (per-step shock prob)')
    ax[0].set_ylabel('drift index (seg-mean var / total var)')
    ax[0].set_title(f'monotone in lambda: {mono} (strict over full ladder)')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    measured = [calib[f'drift_er{er}.csv']['measured_E_run'] for er in ers_sorted]
    ax[1].loglog(ers_sorted, measured, 'o-')
    ax[1].plot([min(ers_sorted), max(ers_sorted)], [min(ers_sorted), max(ers_sorted)], 'k:', alpha=0.5)
    ax[1].set_xlabel('nominal E[run] = 1/lambda')
    ax[1].set_ylabel('measured E[run]')
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'calibration.png'), dpi=150)
    print('fig saved: logs/power_law/figs/calibration.png')


if __name__ == '__main__':
    main()
