#!/usr/bin/env python
"""E3 (part 3): classical PACF/AIC lag-order selection vs measured best_sl.

For each of the 8 datasets, on the standardized? -> NO: on the raw train segment
(sample mean removed implicitly by the ACF), per channel (<=32 sampled, seed=0):
  - ACF up to maxlag=2880 (statsmodels, fft)
  - Levinson-Durbin recursion -> PACF(p) and AR(p) innovation variance sigma2(p)
  - PACF suggested lag = largest p with |pacf(p)| > 1.96/sqrt(N)  (95% band)
  - AIC suggested order = argmin_p N*log(sigma2(p)) + 2p   (Yule-Walker AIC,
    same construction as R ar(aic=TRUE, order.max))
Dataset-level suggestion = median over channels, mapped to the candidate grid
{96,336,720,1440,2880} by (a) nearest on log2 scale and (b) ceiling.
Outputs logs/ehs_fix/e3_pacf_aic.json + .md
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
from statsmodels.tsa.stattools import acf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute

MAXLAG = 2880
MAX_CH = 32
GRID = [96, 336, 720, 1440, 2880]

DATASETS = {
    'ETTh1': (Dataset_ETT_hour, './dataset/ETT-small/', 'ETTh1.csv'),
    'ETTh2': (Dataset_ETT_hour, './dataset/ETT-small/', 'ETTh2.csv'),
    'ETTm1': (Dataset_ETT_minute, './dataset/ETT-small/', 'ETTm1.csv'),
    'ETTm2': (Dataset_ETT_minute, './dataset/ETT-small/', 'ETTm2.csv'),
    'exchange_rate': (Dataset_Custom, './dataset/exchange_rate/', 'exchange_rate.csv'),
    'weather': (Dataset_Custom, './dataset/weather/', 'weather.csv'),
    'electricity': (Dataset_Custom, './dataset/electricity/', 'electricity.csv'),
    'traffic': (Dataset_Custom, './dataset/traffic/', 'traffic.csv'),
}


def levinson(r):
    """Levinson-Durbin on acf r[0..P]; returns pacf[1..P], sigma2[0..P]."""
    P = len(r) - 1
    pacf = np.zeros(P + 1)
    sigma2 = np.zeros(P + 1)
    phi = np.zeros(P + 1)
    v = r[0]
    sigma2[0] = v
    for p in range(1, P + 1):
        acc = r[p] - np.dot(phi[1:p], r[p - 1:0:-1])
        a = acc / v
        phi[1:p] -= a * phi[p - 1:0:-1]
        phi[p] = a
        v *= (1.0 - a * a)
        if v <= 0:
            v = 1e-12
        pacf[p] = a
        sigma2[p] = v
    return pacf, sigma2


def nearest_grid(x):
    lg = np.log2(GRID)
    return GRID[int(np.argmin(np.abs(lg - np.log2(max(x, 1)))))]


def ceil_grid(x):
    for g in GRID:
        if g >= x:
            return g
    return GRID[-1]


def main():
    rows = []
    for name, (cls, root, path) in DATASETS.items():
        ds = cls(SimpleNamespace(augmentation_ratio=0), root_path=root, flag='train',
                 size=[96, 48, 96], features='M', data_path=path,
                 target='OT', scale=False, timeenc=1, freq='h')
        X = ds.data_x  # raw train segment [N, C]
        N, C = X.shape
        rng = np.random.RandomState(0)
        ch = np.arange(C) if C <= MAX_CH else np.sort(rng.choice(C, MAX_CH, replace=False))
        maxlag = min(MAXLAG, N // 2 - 1)
        variants = {'level': X, 'diff': np.diff(X, axis=0)}
        row = {
            'dataset': name, 'N_train': int(N), 'C': int(C), 'n_channels_used': len(ch),
        }
        for vname, V in variants.items():
            pacf_sugs, aic_sugs = [], []
            Nv = V.shape[0]
            for c in ch:
                r = acf(V[:, c], nlags=maxlag, fft=True, missing='conservative')
                pacf, sigma2 = levinson(r)
                band = 1.96 / np.sqrt(Nv)
                sig = np.where(np.abs(pacf[1:]) > band)[0] + 1
                pacf_sugs.append(int(sig.max()) if len(sig) else 0)
                aic = Nv * np.log(sigma2) + 2 * np.arange(maxlag + 1)
                aic_sugs.append(int(np.argmin(aic)))
            row[f'pacf_{vname}_median'] = float(np.median(pacf_sugs))
            row[f'pacf_{vname}_q25'] = float(np.percentile(pacf_sugs, 25))
            row[f'pacf_{vname}_q75'] = float(np.percentile(pacf_sugs, 75))
            row[f'aic_{vname}_median'] = float(np.median(aic_sugs))
            row[f'aic_{vname}_q25'] = float(np.percentile(aic_sugs, 25))
            row[f'aic_{vname}_q75'] = float(np.percentile(aic_sugs, 75))
            row[f'pacf_{vname}_grid'] = nearest_grid(np.median(pacf_sugs))
            row[f'aic_{vname}_grid'] = nearest_grid(np.median(aic_sugs))
            row[f'pacf_{vname}_all'] = pacf_sugs
            row[f'aic_{vname}_all'] = aic_sugs
        rows.append(row)
        print(f"{name:15s} N={N:6d} C={C:4d} | PACF lvl={row['pacf_level_median']:6.0f}->{row['pacf_level_grid']:4d} "
              f"diff={row['pacf_diff_median']:6.0f}->{row['pacf_diff_grid']:4d} | "
              f"AIC lvl={row['aic_level_median']:6.0f}->{row['aic_level_grid']:4d} "
              f"diff={row['aic_diff_median']:6.0f}->{row['aic_diff_grid']:4d}")

    with open('logs/ehs_fix/e3_pacf_aic.json', 'w') as f:
        json.dump(rows, f, indent=2)


if __name__ == '__main__':
    main()
