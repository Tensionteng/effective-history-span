#!/usr/bin/env python
"""Collect EHS v2 EXTENSION lookback results (new datasets).

Parses the last 'mse:..., mae:...' line of every log under logs/ehs_v2_ext/,
aggregates over seeds (mean+-std), writes logs/ehs_v2_ext/SUMMARY.md.
Log naming: {dataset}_{model}_sl{seq_len}_s{seed}.log
"""
import os
import re
import numpy as np

LOG_DIR = 'logs/ehs_v2_ext'
MODELS = ['iTransformer', 'DLinear']
SEQ_LENS = [96, 336, 720, 1440, 2880]
SEEDS = [2021, 2022, 2023]

METRIC_RE = re.compile(r'mse:([\d.eE+-]+),\s*mae:([\d.eE+-]+)')
MAIN_RE = re.compile(r'(.+?)_(iTransformer|DLinear)_sl(\d+)_s(\d+)\.log$')


def last_metric(path):
    val = None
    with open(path, errors='ignore') as f:
        for line in f:
            m = METRIC_RE.search(line)
            if m:
                val = (float(m.group(1)), float(m.group(2)))
    return val


def collect():
    out = {}
    for fn in sorted(os.listdir(LOG_DIR)):
        m = MAIN_RE.match(fn)
        if not m:
            continue
        ds, mo, sl, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        out[(ds, mo, sl, seed)] = last_metric(os.path.join(LOG_DIR, fn))
    return out


def fmt(cell, bold=False):
    if cell is None:
        return '-'
    s = f'{cell[0]:.4f}±{cell[1]:.4f}'
    if cell[2] < 3:
        s += f'[n={cell[2]}]'
    return f'**{s}**' if bold else s


def main():
    rec = collect()
    datasets = sorted({k[0] for k in rec})
    parts = ['# EHS v2 EXTENSION: confounder-controlled lookback on NEW datasets',
             '',
             'Same protocol as ehs_v2: all arms use --max_train_windows = N_min of the sl=2880 arm;',
             'pred_len 96; seeds {2021,2022,2023}; wide(>100ch)=electricity config, narrow=ETT config.',
             '',
             'Cells: mean±std test MSE; **bold** = best seq_len of the row; `[n=k]` < 3 seeds finished.',
             '']
    best_all = {}
    for metric_i, metric_name in [(0, 'MSE'), (1, 'MAE')]:
        for mo in MODELS:
            parts += [f'### Model: {mo} ({metric_name})', '',
                      '| dataset | ' + ' | '.join(f'sl={s}' for s in SEQ_LENS) + ' | best sl |',
                      '|---|---|---|---|---|---|---|']
            for ds in datasets:
                cells = []
                for sl in SEQ_LENS:
                    vals = [rec[(ds, mo, sl, s)][metric_i] for s in SEEDS
                            if rec.get((ds, mo, sl, s)) is not None]
                    cells.append((float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None)
                valid = [(i, c[0]) for i, c in enumerate(cells) if c is not None]
                if not valid:
                    continue
                best_i = min(valid, key=lambda x: x[1])[0]
                if metric_i == 0:
                    best_all.setdefault(mo, {})[ds] = SEQ_LENS[best_i]
                parts.append('| ' + ds + ' | ' + ' | '.join(
                    fmt(c, bold=(i == best_i)) for i, c in enumerate(cells)) + f' | {SEQ_LENS[best_i]} |')
            parts.append('')
    parts += ['### Best seq_len summary (by mean test MSE)', '',
              '| dataset | ' + ' | '.join(MODELS) + ' |', '|---|---|---|']
    for ds in datasets:
        parts.append('| ' + ds + ' | ' + ' | '.join(str(best_all.get(mo, {}).get(ds, '-')) for mo in MODELS) + ' |')
    parts.append('')

    def reason(logpath):
        if not os.path.exists(logpath):
            return 'never started'
        txt = open(logpath, errors='ignore').read()
        if 'Traceback' in txt:
            m = re.findall(r'(\w+(?:Error|Exception))\b', txt)
            return f'failed ({m[-1] if m else "see log"})'
        return 'killed mid-run'

    rows = []
    for ds in datasets:
        for mo in MODELS:
            for sl in SEQ_LENS:
                for s in SEEDS:
                    if rec.get((ds, mo, sl, s)) is None:
                        rows.append((f'{ds} {mo} sl={sl} seed={s}',
                                     reason(os.path.join(LOG_DIR, f'{ds}_{mo}_sl{sl}_s{s}.log'))))
    if rows:
        parts += ['## Missing/failed arms', '']
        for label, why in rows:
            parts.append(f'- {label} — {why}')
        parts.append('')

    out = os.path.join(LOG_DIR, 'SUMMARY.md')
    with open(out, 'w') as f:
        f.write('\n'.join(parts))
    done = sum(1 for v in rec.values() if v)
    print(f'wrote {out}; arms with results: {done} / {len(datasets)*len(MODELS)*len(SEQ_LENS)*len(SEEDS)}')


if __name__ == '__main__':
    main()
