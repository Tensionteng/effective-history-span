#!/usr/bin/env python
"""PEMS04/PEMS08 npz -> TSLib custom CSV (speed channel only, 5-min).

Sources: hf-mirror datasets jimmygao3218/PEMS04, jimmygao3218/PEMS08
(md5-verified identical to Zenodo TrafficDataSets record 7816008).
npz layout: data (T, N, 3) = [flow, occupancy, speed]; we keep speed.
Standard date ranges: PEMS04 = 2018-01-01 + 59d; PEMS08 = 2016-07-01 + 62d.
Outputs: dataset/PEMS04/PEMS04.csv, dataset/PEMS08/PEMS08.csv
"""
import os
import numpy as np
import pandas as pd

SPECS = {
    'PEMS04': ('2018-01-01 00:00:00', 16992, 307),
    'PEMS08': ('2016-07-01 00:00:00', 17856, 170),
}

for name, (start, T, N) in SPECS.items():
    a = np.load(f'dataset/.cache/ehs_ext/{name}.npz')['data']
    assert a.shape == (T, N, 3), a.shape
    speed = a[:, :, 2]
    assert not np.isnan(speed).any()
    dates = pd.date_range(start, periods=T, freq='5min')
    df = pd.DataFrame(speed, columns=[f'ch{i}' for i in range(N)])
    df.insert(0, 'date', dates.strftime('%Y-%m-%d %H:%M:%S'))
    os.makedirs(f'dataset/{name}', exist_ok=True)
    df.to_csv(f'dataset/{name}/{name}.csv', index=False)
    print(f'dataset/{name}/{name}.csv shape={df.shape} '
          f'[{df.date.iloc[0]} .. {df.date.iloc[-1]}]')
