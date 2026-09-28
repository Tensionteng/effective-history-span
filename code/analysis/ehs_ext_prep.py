#!/usr/bin/env python
"""EHS v2 extension: convert downloaded sources to TSLib `custom` CSVs.

Outputs (each: first column `date`, then one column per channel, no NaN):
  dataset/solar/solar.csv           GiftEval solar/10T (LSTNet solar), 137 ch, 10-min, 2006
  dataset/ercot/ercot.csv           autogluon chronos_datasets ercot, 8 ch, hourly, 2004-2021
  dataset/pedestrian/pedestrian.csv monash pedestrian_counts (autogluon), hourly;
                                    sensors restricted to the common contiguous span
"""
import os
import numpy as np
import pandas as pd
import pyarrow.ipc as ipc


def write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    n_nan = int(df.isna().sum().sum())
    print(f'{path}: shape={df.shape} nan={n_nan}')
    print(f'  rows={len(df)} channels={df.shape[1]-1} '
          f'date=[{df.date.iloc[0]} .. {df.date.iloc[-1]}]')


def prep_solar():
    with open('dataset/.cache/ehs_ext/gift_solar/solar_10T.arrow', 'rb') as f:
        try:
            tbl = ipc.RecordBatchFileReader(f).read_all()
        except Exception:
            f.seek(0)
            tbl = ipc.RecordBatchStreamReader(f).read_all()
    d = tbl.to_pandas()
    assert (d['target'].apply(len) == 52560).all()
    dates = pd.date_range('2006-01-01 00:00:00', periods=52560, freq='10min')
    X = np.stack([np.asarray(t, dtype=np.float64) for t in d['target']], axis=1)
    df = pd.DataFrame(X, columns=[f'ch{i}' for i in range(X.shape[1])])
    df.insert(0, 'date', dates.strftime('%Y-%m-%d %H:%M:%S'))
    write_csv(df, 'dataset/solar/solar.csv')


def prep_ercot():
    d = pd.read_parquet('dataset/.cache/ehs_ext/autogluon/ercot.parquet')
    lens = d['target'].apply(len)
    assert (lens == lens.iloc[0]).all()
    ts = pd.DatetimeIndex(pd.to_datetime(pd.Series(d['timestamp'].iloc[0])))
    assert len(ts) == lens.iloc[0]
    X = np.stack([np.asarray(t, dtype=np.float64) for t in d['target']], axis=1)
    df = pd.DataFrame(X, columns=[str(i) for i in d['id']])
    df.insert(0, 'date', ts.strftime('%Y-%m-%d %H:%M:%S'))
    write_csv(df, 'dataset/ercot/ercot.csv')


def prep_pedestrian():
    d = pd.read_parquet('dataset/.cache/ehs_ext/autogluon/monash_pedestrian_counts.parquet')
    starts = d['timestamp'].apply(lambda a: pd.Timestamp(a[0]))
    ends = d['timestamp'].apply(lambda a: pd.Timestamp(a[-1]))
    # common contiguous span: sensors covering [max common start .. min common end]
    span_start = pd.Timestamp('2009-05-01 00:00:00')
    span_end = pd.Timestamp('2018-12-13 18:00:00')  # max end shared by the long sensors
    sel = d[(starts <= span_start) & (ends >= span_end)]
    print(f'pedestrian: {len(sel)}/{len(d)} sensors cover [{span_start} .. {span_end}]')
    n = int((span_end - span_start).total_seconds() // 3600) + 1
    cols = {}
    for _, r in sel.iterrows():
        t = np.asarray(r['target'], dtype=np.float64)
        st = pd.Timestamp(r['timestamp'][0])
        off = int((span_start - st).total_seconds() // 3600)
        seg = t[off:off + n]
        assert len(seg) == n, (r['id'], len(seg), n)
        cols[str(r['id'])] = seg
    dates = pd.date_range(span_start, periods=n, freq='h')
    df = pd.DataFrame(cols)
    df.insert(0, 'date', dates.strftime('%Y-%m-%d %H:%M:%S'))
    write_csv(df, 'dataset/pedestrian/pedestrian.csv')


if __name__ == '__main__':
    prep_solar()
    prep_ercot()
    prep_pedestrian()
