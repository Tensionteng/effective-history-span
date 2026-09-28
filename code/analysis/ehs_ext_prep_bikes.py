#!/usr/bin/env python
"""mexico_city_bikes (autogluon chronos_datasets) -> TSLib custom CSV.

Window [2016-01-01 .. 2022-01-01): stations fully covering it with contiguous
hourly timestamps. Output: dataset/bikes/bikes.csv (last channel renamed OT).
"""
import numpy as np
import pandas as pd

LO = pd.Timestamp('2016-01-01 00:00:00')
HI = pd.Timestamp('2022-01-01 00:00:00')  # exclusive
N = int((HI - LO).total_seconds() // 3600)


def main():
    d = pd.read_parquet('dataset/.cache/ehs_ext/autogluon/mexico_city_bikes.parquet')
    starts = d['timestamp'].apply(lambda a: pd.Timestamp(a[0]))
    ends = d['timestamp'].apply(lambda a: pd.Timestamp(a[-1]))
    sel = d[(starts <= LO) & (ends >= HI - pd.Timedelta(hours=1))]
    cols, dropped = {}, 0
    for _, r in sel.iterrows():
        ts = pd.DatetimeIndex(pd.to_datetime(pd.Series(r['timestamp'])))
        m = np.asarray((ts >= LO) & (ts < HI))
        seg_ts = ts[m]
        t = np.asarray(r['target'], dtype=np.float64)[m]
        if len(t) != N or not (np.diff(seg_ts.values).astype('timedelta64[s]').astype(int) == 3600).all():
            dropped += 1
            continue
        cols[str(r['id'])] = t
    print(f'selected {len(sel)}, kept {len(cols)}, dropped {dropped} (non-contiguous)')
    df = pd.DataFrame(cols)
    dates = pd.date_range(LO, periods=N, freq='h')
    df.insert(0, 'date', dates.strftime('%Y-%m-%d %H:%M:%S'))
    nan = int(df.isna().sum().sum())
    print('NaN:', nan)
    df = df.ffill().bfill()
    cols_l = list(df.columns)
    cols_l[-1] = 'OT'
    df.columns = cols_l
    import os
    os.makedirs('dataset/bikes', exist_ok=True)
    df.to_csv('dataset/bikes/bikes.csv', index=False)
    print(f'wrote dataset/bikes/bikes.csv shape={df.shape}')


if __name__ == '__main__':
    main()
