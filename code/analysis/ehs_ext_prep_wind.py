#!/usr/bin/env python
"""Parse wind_farms_minutely tsf -> 15-min multivariate CSV (TSLib custom).

Keeps the 311 series starting 2019-08-01 00:00:01 with full length 527040 min;
aggregates to 15-min bucket means (buckets with all minutes missing -> NaN,
then column-dropped if >0.5% NaN, remaining gaps ffilled).
Output: dataset/wind_farms/wind_farms.csv
"""
import numpy as np
import pandas as pd

PATH = 'dataset/.cache/ehs_ext/wind_farms_minutely_dataset_with_missing_values.tsf'
START = '2019-08-01 00-00-01'
FULL = 527040  # 366 days (leap) * 1440
B = 15  # minutes per bucket


def main():
    names, cols = [], []
    kept = skipped = 0
    with open(PATH) as f:
        for line in f:
            if line.startswith('@') or line.startswith('#'):
                continue
            name, start, vals = line.rstrip('\n').split(':', 2)
            if start != START:
                skipped += 1
                continue
            v = np.array(vals.split(','))
            if len(v) != FULL:
                skipped += 1
                continue
            a = np.where(v == '?', np.nan, 0.0).astype(np.float64)
            a[v == '?'] = np.nan
            names.append(name)
            cols.append(a)
            kept += 1
    print(f'kept {kept} series, skipped {skipped}')
    X = np.stack(cols, axis=1)  # (FULL, n)
    n_b = FULL // B
    Xb = X[:n_b * B].reshape(n_b, B, X.shape[1])
    with np.errstate(all='ignore'):
        M = np.nanmean(Xb, axis=1)
    nan_frac = np.isnan(M).mean(axis=0)
    print(f'15-min buckets: {n_b}; per-series NaN frac: min={nan_frac.min():.5f} '
          f'median={np.median(nan_frac):.5f} max={nan_frac.max():.5f}')
    keep = nan_frac <= 0.005
    print(f'dropping {int((~keep).sum())} series with >0.5% NaN buckets')
    M = M[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    df = pd.DataFrame(M, columns=names)
    dates = pd.date_range('2019-08-01 00:00:00', periods=n_b, freq='15min')
    df.insert(0, 'date', dates.strftime('%Y-%m-%d %H:%M:%S'))
    before = int(df.isna().sum().sum())
    df = df.ffill().bfill()
    print(f'NaN buckets before fill: {before}, after: {int(df.isna().sum().sum())}')
    import os
    os.makedirs('dataset/wind_farms', exist_ok=True)
    df.to_csv('dataset/wind_farms/wind_farms.csv', index=False)
    print(f'wrote dataset/wind_farms/wind_farms.csv shape={df.shape}')


if __name__ == '__main__':
    main()
