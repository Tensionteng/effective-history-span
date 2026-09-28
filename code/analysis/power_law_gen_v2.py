#!/usr/bin/env python
"""v2 hardening of the W* ~ lambda ladder: 6 -> 10 tiers, middle-dense.

Ladder: E[run] in {50, 150, 250, 400, 650, 1000, 1700, 2500, 4000, 6000}
(v1 tiers 50/150/400/1000/2500/6000 kept; new middle tiers 250/650/1700/4000).
New CSVs are generated with the SAME sigma=12 high-noise design and seed=7 as v1
(analysis/power_law_gen.py), so v1 CSVs stay byte-identical and are reused untouched.
The stale sigma=1 files drift_er{100,220,460,960,2000}.csv (v1_lowsnr archive) are
NOT part of this ladder.

Outputs: new CSVs under dataset/power_law/; calibration (30 realizations x 4 channels
per tier + drift of the actual train CSVs + monotonicity check) into
logs/power_law/v2/calibration.json and logs/power_law/v2/figs/calibration.png.

Run: .venv/bin/python analysis/power_law_gen_v2.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from power_law_gen import gen_series, drift_index, write_csv, T, C  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'dataset', 'power_law')
LOG = os.path.join(ROOT, 'logs', 'power_law', 'v2')
FIG = os.path.join(LOG, 'figs')

TRAIN_T = 17500  # Dataset_Custom 70% train split of T=25000
ER_LADDER = [50, 150, 250, 400, 650, 1000, 1700, 2500, 4000, 6000]
ER_NEW = [250, 650, 1700, 4000]
N_REPL = 30


def main():
    os.makedirs(FIG, exist_ok=True)
    calib = {}

    # 1. new middle-tier CSVs (skip if already written; v1 CSVs never touched)
    metas = {}
    for er in ER_LADDER:
        lam = 1.0 / er
        X, meta = gen_series(T, lam, C, False, seed=7)  # deterministic: reproduces v1 CSVs
        metas[er] = meta
        name = f'drift_er{er}.csv'
        path = os.path.join(OUT, name)
        if er in ER_NEW and not os.path.exists(path):
            write_csv(name, X)
            print(f'wrote {name}: lam={lam:.5f} measured_E[r]={meta["measured_E_run"]:.0f}',
                  flush=True)
        elif er in ER_NEW:
            print(f'{name} exists, generation skipped', flush=True)

    # reproducibility guard: regenerated seed=7 series must equal the v1 CSV on disk
    df50 = pd.read_csv(os.path.join(OUT, 'drift_er50.csv'))
    X50, _ = gen_series(T, 1.0 / 50, C, False, seed=7)
    same = np.allclose(df50[['ch0', 'ch1', 'ch2', 'OT']].values, X50, atol=1e-8)
    print(f'reproducibility check drift_er50 (seed=7 regeneration == v1 CSV): {same}',
          flush=True)
    calib['_repro_check_er50'] = bool(same)

    # 2. drift of the actual training CSVs (train split only)
    for er in ER_LADDER:
        name = f'drift_er{er}.csv'
        df = pd.read_csv(os.path.join(OUT, name))
        Xtr = df.iloc[:TRAIN_T][['ch0', 'ch1', 'ch2', 'OT']].values
        drifts = [drift_index(Xtr[:, c]) for c in range(Xtr.shape[1])]
        calib[name] = dict(lam=1.0 / er, nominal_E_run=er,
                           measured_E_run=metas[er]['measured_E_run'],
                           n_shocks=metas[er]['n_shocks'], periodic=False,
                           drift_per_channel=drifts, drift_mean=float(np.mean(drifts)),
                           drift_std=float(np.std(drifts)))
        print(f'{name}: lam={1 / er:.5f} measured_E[r]={metas[er]["measured_E_run"]:.0f} '
              f'drift={np.mean(drifts):.4f}±{np.std(drifts):.4f}', flush=True)

    # 3. multi-realization calibration curve over all 10 tiers
    curve = {}
    for er in ER_LADDER:
        lam = 1.0 / er
        vals, meas = [], []
        for rep in range(N_REPL):
            X, meta = gen_series(T, lam, C, False, seed=1000 + rep)
            vals += [drift_index(X[:TRAIN_T, c]) for c in range(C)]
            meas.append(meta['measured_E_run'])
        curve[er] = dict(lam=lam, drift_mean=float(np.mean(vals)),
                         drift_std=float(np.std(vals)),
                         drift_sem=float(np.std(vals) / np.sqrt(len(vals))), n=len(vals),
                         measured_E_run=float(np.mean(meas)))
        print(f'calib er={er}: drift={np.mean(vals):.4f}±{np.std(vals) / np.sqrt(len(vals)):.4f} '
              f'(n={len(vals)}) measured_E[r]={np.mean(meas):.0f}', flush=True)
    calib['_curve'] = {str(k): v for k, v in curve.items()}

    ers_sorted = sorted(ER_LADDER)  # ascending E[r] = descending lambda
    drifts = [curve[e]['drift_mean'] for e in ers_sorted]
    mono = all(drifts[i] >= drifts[i + 1] for i in range(len(drifts) - 1))
    print(f'drift_index monotone non-increasing along ascending E[r]: {mono}', flush=True)
    calib['_monotone'] = bool(mono)
    with open(os.path.join(LOG, 'calibration.json'), 'w') as f:
        json.dump(calib, f, indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    lams = np.array([curve[e]['lam'] for e in ers_sorted])
    ds = np.array(drifts)
    sems = np.array([curve[e]['drift_sem'] for e in ers_sorted])
    order = np.argsort(lams)
    ax[0].errorbar(lams[order], ds[order], yerr=sems[order], fmt='o-', capsize=3,
                   label=f'pure drift ({N_REPL} realizations)')
    dser = [calib[f'drift_er{er}.csv']['drift_mean'] for er in ers_sorted]
    ax[0].loglog(lams[order], np.array(dser)[order], 'x', ms=8, label='actual training CSV')
    ax[0].set_xlabel('lambda (per-step shock prob)')
    ax[0].set_ylabel('drift index (seg-mean var / total var)')
    ax[0].set_title(f'v2 10-tier ladder, monotone in lambda: {mono}')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    measured = [curve[e]['measured_E_run'] for e in ers_sorted]
    ax[1].loglog(ers_sorted, measured, 'o-')
    ax[1].plot([min(ers_sorted), max(ers_sorted)], [min(ers_sorted), max(ers_sorted)],
               'k:', alpha=0.5)
    ax[1].set_xlabel('nominal E[run] = 1/lambda')
    ax[1].set_ylabel('measured E[run]')
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'calibration.png'), dpi=150)
    print('fig saved: logs/power_law/v2/figs/calibration.png', flush=True)


if __name__ == '__main__':
    main()
