#!/usr/bin/env python
"""Two-axis EHS statistics on the GIFT-Eval task cache (analysis/ehs_stats.py 口径).

Per task, per sampled series, on the TRAIN segment only (series[:-W*P], no test
leakage), first <=12000 steps (parity with ehs_predict_v2), NaN linearly
interpolated (stats only -- inference masks NaN natively):
  periodicity_dt     spectral peak ratio after linear detrend
  drift              variance of 20-segment means / total variance        (axis 1)
  long_acf           ACF at lag 25% of length (raw series)
  calendar_acf_diff  mean ACF at calendar lags, detrended+differenced     (axis 2)
  long_acf_diff      ACF at 25% lag, detrended+differenced                (axis 2)

Output: logs/gift_ext/stats.csv (mean/std over series per task)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.ehs_stats import acf_at, detrend, drift_index, long_acf, periodicity_detrended

CACHE = 'logs/gift_ext/data/cache'
OUT = 'logs/gift_ext/stats.csv'
MAX_LEN = 12000


def interp_nan(x):
    if not np.isnan(x).any():
        return x
    s = pd.Series(x).interpolate(limit_direction='both').ffill().bfill()
    return s.to_numpy(dtype=np.float64)


def calendar_acf_diff(x, periods):
    xd = np.diff(detrend(x - x.mean()))
    return float(np.nanmean([acf_at(xd, p) for p in periods])) if periods else np.nan


def long_acf_diff(x, lag_frac=0.25):
    xd = np.diff(detrend(x - x.mean()))
    return acf_at(xd, max(1, int(len(xd) * lag_frac)))


def series_stats(x, periods):
    return {'periodicity_dt': periodicity_detrended(x),
            'drift': drift_index(x),
            'long_acf': long_acf(x),
            'calendar_acf_diff': calendar_acf_diff(x, periods),
            'long_acf_diff': long_acf_diff(x)}


def main():
    meta = json.load(open('logs/gift_ext/tasks.json'))
    rows = []
    for task, m in meta.items():
        df = pd.read_parquet(os.path.join(CACHE, f'{task}.parquet'))
        P, W, periods = m['pred_len'], m['windows'], m['periods']
        per = []
        for r in df.itertuples():
            v = np.asarray(r.target, dtype=np.float64)
            train = interp_nan(v[:len(v) - W * P])[:MAX_LEN]
            need = (2 * max(periods) + 10) if periods else 100
            if len(train) < need:
                continue
            per.append(series_stats(train, periods))
        row = {'task': task, 'domain': m['domain'], 'freq': m['freq'],
               'pred_len': P, 'windows': W, 'n_series': len(per),
               'periods': json.dumps(periods), 'season': m['season'],
               'context_capped': m['context_capped']}
        for k in ['periodicity_dt', 'drift', 'long_acf', 'calendar_acf_diff', 'long_acf_diff']:
            v = np.array([p[k] for p in per], dtype=float)
            row[f'{k}_mean'] = float(np.nanmean(v))
            row[f'{k}_std'] = float(np.nanstd(v))
        rows.append(row)
        print(f"{task:28} drift={row['drift_mean']:+.4f} cal={row['calendar_acf_diff_mean']:+.4f} "
              f"longacf_d={row['long_acf_diff_mean']:+.4f} per_dt={row['periodicity_dt_mean']:.4f}",
          flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
