#!/usr/bin/env python
"""E6 coverage-fill verdict: PatchTST/TimesNet x {electricity,traffic} lookback
curves after filling the OOM-missing long-window arms.

Parses logs/ehs_v2/{dataset}_{model}_sl{sl}_s{seed}.log for the four
(model, dataset) combos, aggregates mean+-std over available seeds, and asks:
does the best seq_len stay at the short windows (96/336) once the long arms
are covered? Writes logs/ehs_fix/e6_fill_verdict.{json,md}.
"""
import json
import os
import re

import numpy as np

LOG_DIR = 'logs/ehs_v2'
MODELS = ['PatchTST', 'TimesNet']
DATASETS = ['electricity', 'traffic']
SEQ_LENS = [96, 336, 720, 1440, 2880]
SEEDS = [2021, 2022, 2023]
SHORT = [96, 336]
LONG = [720, 1440, 2880]

METRIC_RE = re.compile(r'mse:([\d.eE+-]+),\s*mae:([\d.eE+-]+)')


def last_metric(path):
    val = None
    with open(path, errors='ignore') as f:
        for line in f:
            m = METRIC_RE.search(line)
            if m:
                val = (float(m.group(1)), float(m.group(2)))
    return val


def main():
    cells = {}  # (ds, model, sl) -> dict(mse=[(seed,val)...], mae=[...])
    for ds in DATASETS:
        for mo in MODELS:
            for sl in SEQ_LENS:
                mses, maes = [], []
                for s in SEEDS:
                    p = os.path.join(LOG_DIR, f'{ds}_{mo}_sl{sl}_s{s}.log')
                    if not os.path.exists(p):
                        continue
                    v = last_metric(p)
                    if v:
                        mses.append((s, v[0]))
                        maes.append((s, v[1]))
                cells[(ds, mo, sl)] = {'mse': mses, 'mae': maes}

    def agg(ds, mo, sl, key):
        vals = [v for _, v in cells[(ds, mo, sl)][key]]
        if not vals:
            return None
        return (float(np.mean(vals)), float(np.std(vals)), len(vals))

    verdict = {}
    for key in ['mse', 'mae']:
        verdict[key] = {}
        for ds in DATASETS:
            for mo in MODELS:
                row = {sl: agg(ds, mo, sl, key) for sl in SEQ_LENS}
                have = {sl: c for sl, c in row.items() if c}
                if not have:
                    continue
                best_all = min(have, key=lambda sl: have[sl][0])
                short_have = {sl: c for sl, c in have.items() if sl in SHORT}
                long_have = {sl: c for sl, c in have.items() if sl in LONG}
                best_short = min(short_have, key=lambda sl: short_have[sl][0]) if short_have else None
                best_long = min(long_have, key=lambda sl: long_have[sl][0]) if long_have else None
                verdict[key][f'{ds}/{mo}'] = {
                    'cells': {str(sl): c for sl, c in row.items()},
                    'best_sl': best_all,
                    'best_short': best_short,
                    'best_long': best_long,
                    'long_beats_short': bool(best_long and best_short and
                                             have[best_long][0] < have[best_short][0]),
                    'long_minus_short': (have[best_long][0] - have[best_short][0]
                                         if best_long and best_short else None),
                }

    out = {'cells': {f'{ds}/{mo}/sl{sl}': cells[(ds, mo, sl)]
                     for ds in DATASETS for mo in MODELS for sl in SEQ_LENS},
           'verdict': verdict}
    os.makedirs('logs/ehs_fix', exist_ok=True)
    with open('logs/ehs_fix/e6_fill_verdict.json', 'w') as f:
        json.dump(out, f, indent=1)

    lines = []
    for key, label in [('mse', 'MSE'), ('mae', 'MAE')]:
        lines += [f'### {label}', '',
                  '| row | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | best sl | best short | best long | Δ(long-short) |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for ds in DATASETS:
            for mo in MODELS:
                v = verdict[key].get(f'{ds}/{mo}')
                if not v:
                    continue
                cs = []
                for sl in SEQ_LENS:
                    c = v['cells'].get(str(sl))
                    if c is None:
                        cs.append('-')
                    else:
                        s = f'{c[0]:.4f}±{c[1]:.4f}'
                        if c[2] < 3:
                            s += f'[n={c[2]}]'
                        cs.append(f'**{s}**' if sl == v['best_sl'] else s)
                d = v['long_minus_short']
                lines.append('| %s/%s | %s | %s | %s | %s |' % (ds, mo, *cs[:4])
                             + f' {cs[4]} | {v["best_sl"]} | {v["best_short"]} | {v["best_long"]} | '
                             + ('-' if d is None else f'{d:+.4f}') + ' |')
        lines.append('')
    with open('logs/ehs_fix/e6_fill_verdict.md', 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print('wrote logs/ehs_fix/e6_fill_verdict.{json,md}')


if __name__ == '__main__':
    main()
