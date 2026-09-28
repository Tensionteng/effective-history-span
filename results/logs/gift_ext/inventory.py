#!/usr/bin/env python
"""Inspect downloaded GiftEval arrow files -> logs/gift_ext/inventory.csv.

GIFT-Eval protocol (github.com/SalesforceAIResearch/gift-eval src/gift_eval/data.py):
  prediction_length = term.multiplier * PRED_LENGTH_MAP[freq]  (M4 uses M4 map)
  windows           = 1 for m4 else min(20, max(1, ceil(0.1 * min_series_length / P)))
  test              = last windows*P steps, rolling windows with distance=P
We use Term.SHORT (multiplier 1) throughout.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RAW = 'logs/gift_ext/data/raw'
OUT = 'logs/gift_ext/inventory.csv'

PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
M4_PRED_LENGTH_MAP = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
TEST_SPLIT, MAX_WINDOW = 0.1, 20

# calendar periods (steps) for the two-axis stats + seasonal-naive season m
PERIODS = {'Y': [], 'A': [], 'Q': [4], 'M': [12], 'W': [4, 52], 'D': [7, 30],
           'H': [24, 168], '15T': [96, 672], '10T': [144, 1008],
           '5T': [288, 2016], 'T': [1440], 'S': [60]}
SEASON = {'Y': 1, 'A': 1, 'Q': 4, 'M': 12, 'W': 52, 'D': 7,
          'H': 24, '15T': 96, '10T': 144, '5T': 288, 'T': 1440, 'S': 60}

DOMAIN = {
    'LOOP_SEATTLE': 'transport', 'M_DENSE': 'transport', 'SZ_TAXI': 'transport',
    'bitbrains_fast_storage': 'cloud', 'bitbrains_rnd': 'cloud',
    'bizitobs_application': 'cloud', 'bizitobs_l2c': 'cloud', 'bizitobs_service': 'cloud',
    'car_parts_with_missing': 'sales', 'restaurant': 'sales', 'hierarchical_sales': 'sales',
    'covid_deaths': 'health', 'hospital': 'health', 'us_births': 'demo',
    'electricity': 'energy', 'ett1': 'energy', 'ett2': 'energy', 'solar': 'energy',
    'jena_weather': 'nature', 'kdd_cup_2018_with_missing': 'nature',
    'saugeenday': 'nature', 'temperature_rain_with_missing': 'nature',
    'm4_daily': 'econ', 'm4_hourly': 'econ', 'm4_monthly': 'econ',
    'm4_quarterly': 'econ', 'm4_weekly': 'econ', 'm4_yearly': 'econ',
}


def freq_base(freq):
    """'15T' -> ('15T'), 'H'->'H'; pandas-ish uppercase."""
    f = str(freq).upper()
    return f


def main():
    import datasets
    datasets.logging.set_verbosity_error()
    rows = []
    for root, _, files in os.walk(RAW):
        for fn in sorted(files):
            if not fn.endswith('.arrow'):
                continue
            path = os.path.join(root, fn)
            config = os.path.relpath(root, RAW)
            name = config.split('/')[0]
            try:
                ds = datasets.Dataset.from_file(path)
            except Exception as e:
                rows.append({'config': config, 'error': str(e)[:120]})
                continue
            freq = freq_base(ds[0]['freq'])
            t0 = np.asarray(ds[0]['target'], dtype=object)
            target_dim = len(t0) if isinstance(ds[0]['target'][0], (list, tuple)) else 1
            n = len(ds)
            # series lengths (all series; for multivariate count channel length)
            lens, nan_frac = [], []
            for i in range(len(ds)):
                t = np.asarray(ds[i]['target'], dtype=np.float32)
                if t.ndim > 1:
                    lens.append(t.shape[1])
                    nan_frac.append(float(np.isnan(t).mean()))
                else:
                    lens.append(len(t))
                    nan_frac.append(float(np.isnan(t).mean()))
            lens = np.array(lens)
            min_len = int(lens.min())
            pred_map = M4_PRED_LENGTH_MAP if 'm4' in name else PRED_LENGTH_MAP
            P = pred_map.get(freq, PRED_LENGTH_MAP.get(freq[-1], 48))
            W = 1 if 'm4' in name else min(MAX_WINDOW, max(1, math.ceil(TEST_SPLIT * min_len / P)))
            need = 2048 + W * P
            rows.append({
                'config': config, 'name': name, 'domain': DOMAIN.get(name, '?'),
                'freq': freq, 'n_series': n, 'target_dim': target_dim,
                'n_univariate': n * target_dim,
                'min_len': min_len, 'med_len': int(np.median(lens)),
                'max_len': int(lens.max()),
                'nan_frac': float(np.mean(nan_frac)),
                'pred_len': P, 'windows': W,
                'len_ok_2048': bool(min_len >= need),
                'len_ok_1440': bool(min_len >= 1440 + W * P),
                'periods': json.dumps(PERIODS.get(freq, [])),
                'season': SEASON.get(freq, 1),
                'size_mb': round(os.path.getsize(path) / 1e6, 1),
            })
            print(config, rows[-1]['freq'], 'min_len', min_len, 'P', P, 'W', W,
                  'ok2048', rows[-1]['len_ok_2048'], flush=True)
    df = pd.DataFrame(rows).sort_values('config')
    df.to_csv(OUT, index=False)
    print(f'\nwrote {OUT} ({len(df)} configs)')
    print(df[['config', 'freq', 'n_univariate', 'min_len', 'pred_len', 'windows',
              'len_ok_2048', 'nan_frac']].to_string(index=False))


if __name__ == '__main__':
    main()
