#!/usr/bin/env python
"""Recompute scan_summary.csv from saved scan/*.npz (no re-inference).
Uses the current metrics_block from scan.py (NaN-target-masked metrics)."""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan import metrics_block, LS, OUT_DIR, SUMMARY


def main():
    meta = json.load(open('logs/gift_ext/tasks.json'))
    rows = []
    for task in meta:
        path = os.path.join(OUT_DIR, f'{task}.npz')
        if not os.path.exists(path):
            print(f'SKIP {task}: no npz', flush=True)
            continue
        z = np.load(path, allow_pickle=True)
        quants, tgts, snaives, scales = z['quants'], z['target'], z['snaive'], z['mase_scale']
        nS, W, P = tgts.shape
        m = meta[task]
        assert P == m['pred_len'] and W == m['windows'], f'{task}: protocol mismatch'
        for li, L in enumerate(z['Ls']):
            mb = metrics_block(quants[li].reshape(-1, 9, P), tgts.reshape(-1, P),
                               snaives.reshape(-1, P), np.repeat(scales, W))
            rows.append([task, m['domain'], m['freq'], P, W, nS, int(L), min(int(L), 2048),
                         f"{mb['crps']:.6f}", f"{mb['mase']:.6f}" if np.isfinite(mb['mase']) else 'nan',
                         f"{mb['mse']:.6f}", f"{mb['mae']:.6f}",
                         f"{mb['nmse']:.6f}" if np.isfinite(mb['nmse']) else 'nan',
                         mb['n_items'], f"{mb['nan_step_frac']:.4f}"])
    with open(SUMMARY, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['task', 'domain', 'freq', 'pred_len', 'windows', 'n_series',
                     'L', 'L_eff', 'crps', 'mase', 'mse', 'mae', 'nmse', 'n_items',
                     'nan_step_frac'])
        wr.writerows(rows)
    print(f'wrote {SUMMARY} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
