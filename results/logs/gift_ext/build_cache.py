#!/usr/bin/env python
"""Build unified parquet cache for selected GIFT-Eval configs.

Selection rule: min_series_length >= 2048 + windows*pred_len (full L coverage,
the L=2880 arm is capped at bolt's context_length=2048 anyway), plus a small
number of flagged context-capped tasks for frequency/domain diversity
(hierarchical_sales/D: min input length 1615, only the 2880 arm is capped).

Per task we keep up to --max-series univariate series (multivariate datasets are
expanded per channel, following GIFT-Eval's MultivariateToUnivariate), sampled
deterministically (seed 0). NaNs are kept as-is for inference (bolt masks them);
the stats step interpolates separately.

Outputs:
  logs/gift_ext/data/cache/<task>.parquet  (task, series_id, freq, start, target)
  logs/gift_ext/tasks.json                 (protocol per task)
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

RAW = 'logs/gift_ext/data/raw'
CACHE = 'logs/gift_ext/data/cache'
TASKS_JSON = 'logs/gift_ext/tasks.json'

PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
M4_PRED_LENGTH_MAP = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
TEST_SPLIT, MAX_WINDOW = 0.1, 20
PERIODS = {'Y': [], 'A': [], 'Q': [4], 'M': [12], 'W': [4, 52], 'D': [7, 30],
           'H': [24, 168], '15T': [96, 672], '10T': [144, 1008],
           '5T': [288, 2016], 'T': [1440], 'S': [60], '10S': [360]}
SEASON = {'Y': 1, 'A': 1, 'Q': 4, 'M': 12, 'W': 52, 'D': 7,
          'H': 24, '15T': 96, '10T': 144, '5T': 288, 'T': 1440, 'S': 60, '10S': 360}
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

SELECT = [
    # full L coverage (min_len >= 2048 + W*P)
    'LOOP_SEATTLE/5T', 'LOOP_SEATTLE/H', 'M_DENSE/H', 'SZ_TAXI/15T',
    'bitbrains_fast_storage/5T', 'bitbrains_rnd/5T',
    'bizitobs_application', 'bizitobs_l2c/5T', 'bizitobs_l2c/H', 'bizitobs_service',
    'electricity/15T', 'electricity/H',
    'ett1/15T', 'ett1/H', 'ett2/15T', 'ett2/H',
    'jena_weather/10T', 'jena_weather/H', 'kdd_cup_2018_with_missing/H',
    'saugeenday/D', 'saugeenday/W', 'solar/10T', 'solar/H', 'us_births/D',
    # flagged: context-capped (only the L=2880 arm truncates below 2048)
    'hierarchical_sales/D',
]


def norm_freq(freq):
    f = str(freq).split('-')[0].upper()
    return {'MIN': 'T', 'SEC': 'S'}.get(f, f)


def task_name(config):
    return config.replace('/', '_')


def main():
    import datasets
    datasets.logging.set_verbosity_error()
    os.makedirs(CACHE, exist_ok=True)
    max_series = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    rng = np.random.default_rng(0)
    meta = {}
    for config in SELECT:
        path = os.path.join(RAW, config, 'data-00000-of-00001.arrow')
        ds = datasets.Dataset.from_file(path)
        freq = norm_freq(ds[0]['freq'])
        is_m4 = 'm4' in config
        P = (M4_PRED_LENGTH_MAP if is_m4 else PRED_LENGTH_MAP)[freq if freq in (
            M4_PRED_LENGTH_MAP if is_m4 else PRED_LENGTH_MAP) else freq[-1]]
        # expand to univariate series
        series, lens = [], []
        for i in range(len(ds)):
            t = ds[i]['target']
            iid = ds[i]['item_id']
            if isinstance(t[0], (list, tuple)):
                for c, ch in enumerate(t):
                    series.append((f'{iid}_dim{c}', np.asarray(ch, dtype=np.float32)))
            else:
                series.append((str(iid), np.asarray(t, dtype=np.float32)))
        lens = np.array([len(s[1]) for s in series])
        min_len = int(lens.min())
        W = 1 if is_m4 else min(MAX_WINDOW, max(1, math.ceil(TEST_SPLIT * min_len / P)))
        n_all = len(series)
        if n_all > max_series:
            idx = np.sort(rng.choice(n_all, max_series, replace=False))
            series = [series[i] for i in idx]
        capped = bool(min_len < 2048 + W * P)
        min_input = int(min((len(v) for _, v in series))) - W * P
        rows = [(task_name(config), sid, freq, str(ds[0]['start']), v)
                for sid, v in series]
        df = pd.DataFrame(rows, columns=['task', 'series_id', 'freq', 'start', 'target'])
        out = os.path.join(CACHE, f'{task_name(config)}.parquet')
        df.to_parquet(out, index=False)
        meta[task_name(config)] = {
            'config': config, 'name': config.split('/')[0], 'domain': DOMAIN.get(config.split('/')[0], '?'),
            'freq': freq, 'pred_len': P, 'windows': W, 'term': 'SHORT',
            'n_series_total': int(n_all), 'n_series_used': len(series),
            'min_len': int(min((len(v) for _, v in series))),
            'med_len': int(np.median([len(v) for _, v in series])),
            'min_input_len': min_input,
            'context_capped': capped, 'season': SEASON.get(freq, 1),
            'periods': PERIODS.get(freq, []),
            'nan_frac': float(np.mean([np.isnan(v).mean() for _, v in series])),
        }
        print(f'{task_name(config):28} freq={freq:4} P={P:3} W={W:2} series={len(series):3}/{n_all:<5} '
              f'min_len={meta[task_name(config)]["min_len"]:6} capped={capped}', flush=True)
    with open(TASKS_JSON, 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'\nwrote {TASKS_JSON} + {len(meta)} parquets to {CACHE}')


if __name__ == '__main__':
    main()
