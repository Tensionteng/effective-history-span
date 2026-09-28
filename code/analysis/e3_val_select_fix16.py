#!/usr/bin/env python
"""E3-fix (review v5 C007): recompute val/AIC/PACF selector rows over the FULL
16-cell main suite (8 datasets x {iTransformer, DLinear}), fixing the filename
regex that silently dropped exchange_rate (underscore) in e3_val_select.py.

Outputs logs/ehs_fix/e3_fix16.{json,md}.
"""
import json
import os
import re
from collections import defaultdict

LOGDIR = 'logs/ehs_v2'
PACF_AIC = 'logs/ehs_fix/e3_pacf_aic.json'
OUT = 'logs/ehs_fix/e3_fix16.json'
MD = 'logs/ehs_fix/e3_fix16.md'
DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather', 'electricity', 'traffic']
MODELS = ['iTransformer', 'DLinear']

# dataset/model names may contain underscores; model is the token right before _sl
re_name = re.compile(r'^(.+)_([A-Za-z]+)_sl(\d+)_s(\d+)\.log$')
re_vali = re.compile(r'Vali Loss: ([\d.]+)')
re_mse = re.compile(r'^mse:([\d.eE+-]+)', re.M)


def parse_log(path):
    with open(path, errors='ignore') as f:
        txt = f.read()
    valis = [float(v) for v in re_vali.findall(txt)]
    mses = re_mse.findall(txt)
    return (min(valis) if valis else None), (float(mses[-1]) if mses else None)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    data = defaultdict(lambda: defaultdict(dict))  # data[ds][model][sl] = [(vali,mse), ...]
    for fn in sorted(os.listdir(LOGDIR)):
        m = re_name.match(fn)
        if not m:
            continue
        ds, model, sl = m.group(1), m.group(2), int(m.group(3))
        if ds not in DATASETS or model not in MODELS:
            continue
        vali, mse = parse_log(os.path.join(LOGDIR, fn))
        if vali is None or mse is None:
            continue
        data[ds][model].setdefault(sl, []).append((vali, mse))

    cells = []
    for ds in DATASETS:
        for model in MODELS:
            arms = data.get(ds, {}).get(model, {})
            if not arms:
                continue
            mean_vali = {sl: mean([v[0] for v in vs]) for sl, vs in arms.items()}
            mean_mse = {sl: mean([v[1] for v in vs]) for sl, vs in arms.items()}
            val_sl = min(mean_vali, key=mean_vali.get)
            ora_sl = min(mean_mse, key=mean_mse.get)
            cells.append({'dataset': ds, 'model': model,
                          'val_sl': val_sl, 'oracle_sl': ora_sl,
                          'mse_oracle': mean_mse[ora_sl],
                          'mse_val': mean_mse[val_sl],
                          'val_hit': val_sl == ora_sl,
                          'val_rel_regret_pct': 100 * (mean_mse[val_sl] - mean_mse[ora_sl]) / mean_mse[ora_sl],
                          'mse_by_sl': {str(k): v for k, v in sorted(mean_mse.items())}})

    pa = {x['dataset']: x for x in json.load(open(PACF_AIC))}
    for c in cells:
        g = pa[c['dataset']]
        for key, grid in (('aic', g['aic_level_grid']), ('pacf', g['pacf_level_grid'])):
            mm = c['mse_by_sl']
            hit = str(grid) in mm and grid == c['oracle_sl']
            regret = (100 * (mm[str(grid)] - c['mse_oracle']) / c['mse_oracle']) if str(grid) in mm else None
            c[f'{key}_grid'] = grid
            c[f'{key}_hit'] = hit
            c[f'{key}_rel_regret_pct'] = regret

    def agg(key):
        hits = sum(1 for c in cells if c[f'{key}_hit'])
        regs = [c[f'{key}_rel_regret_pct'] for c in cells if c[f'{key}_rel_regret_pct'] is not None]
        return hits, len(cells), sum(regs) / len(regs), sorted(regs)[len(regs) // 2]

    summary = {}
    for key in ('val', 'aic', 'pacf'):
        h, n, mr, med = agg(key)
        summary[key] = {'exact_hits': h, 'n_cells': n, 'mean_rel_regret_pct': round(mr, 2), 'median_rel_regret_pct': round(med, 2)}

    json.dump({'cells': cells, 'summary': summary}, open(OUT, 'w'), indent=2)
    lines = ['| dataset | model | oracle | val_sl | val hit | val rel% | aic grid | aic hit | pacf grid | pacf rel% |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for c in cells:
        pacf_str = ('%+.1f' % c['pacf_rel_regret_pct']) if c['pacf_rel_regret_pct'] is not None else 'NA'
        lines.append(f"| {c['dataset']} | {c['model']} | {c['oracle_sl']} | {c['val_sl']} | "
                     f"{'Y' if c['val_hit'] else 'n'} | {c['val_rel_regret_pct']:+.2f} | {c['aic_grid']} | "
                     f"{'Y' if c['aic_hit'] else 'n'} | {c['pacf_grid']} | {pacf_str} |")
    lines += ['', json.dumps(summary, indent=2)]
    open(MD, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
