#!/usr/bin/env python
"""E3 verdict: three lookback selectors vs test-oracle on the trained matrix.

Selectors per (dataset, model), model in {iTransformer, DLinear}:
  val  : argmin mean best-vali-loss (analysis/e3_val_select.py output)
  stat : task-B k=3 LOO ridge prediction tier (parsed from logs/ehs_final/taskB_predict.txt)
  aic  : dataset-level AIC AR-order median -> nearest grid (analysis/e3_pacf_aic.py; level variant)
  pacf : dataset-level PACF last-significant-lag -> nearest grid
Regret = test mse at selected sl - test mse at oracle sl (mean over seeds,
from the ehs_v2 logs). Outputs logs/ehs_fix/e3_verdict.json + .md
"""
import json
import re

VAL = json.load(open('logs/ehs_fix/e3_val_select_trained.json'))
PACF = {r['dataset']: r for r in json.load(open('logs/ehs_fix/e3_pacf_aic.json'))}

# --- parse task-B k=3 LOO predicted tiers ---
txt = open('logs/ehs_final/taskB_predict.txt').read()
k3_block = txt.split('k=3:')[1]
pred = {}
for m in re.finditer(r'^\s+(\S+)\s+(iTransformer|DLinear)\s+true=\s*(\d+)\s+pred_tier=\s*(\d+)',
                     k3_block, re.M):
    pred[(m.group(1), m.group(2))] = int(m.group(4))

# --- mean test mse per (dataset, model, sl) from the val-select parse ---
mse = {}
for r in VAL:
    if r['model'] not in ('iTransformer', 'DLinear'):
        continue
    mse[(r['dataset'], r['model'])] = {int(k): v for k, v in r['mean_mse'].items()}

rows = []
for (ds, model), msl in sorted(mse.items()):
    ora_sl = min(msl, key=msl.get)
    val_sl = next(r['val_sl'] for r in VAL if r['dataset'] == ds and r['model'] == model)
    stat_sl = pred[(ds, model)]
    aic_sl = PACF[ds]['aic_level_grid']
    pacf_sl = PACF[ds]['pacf_level_grid']
    row = {'dataset': ds, 'model': model, 'oracle_sl': ora_sl, 'oracle_mse': msl[ora_sl]}
    for name, sl in [('val', val_sl), ('stat', stat_sl), ('aic', aic_sl), ('pacf', pacf_sl)]:
        row[f'{name}_sl'] = sl
        row[f'{name}_mse'] = msl[sl]
        row[f'{name}_regret'] = msl[sl] - msl[ora_sl]
        row[f'{name}_hit'] = sl == ora_sl
    rows.append(row)

with open('logs/ehs_fix/e3_verdict.json', 'w') as f:
    json.dump(rows, f, indent=2)

hdr = '| dataset | model | oracle sl | val sl | stat sl | AIC sl | PACF sl | mse(oracle) | regret val | regret stat | regret AIC | regret PACF |'
lines = [hdr, '|---|---|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    lines.append(f"| {r['dataset']} | {r['model']} | {r['oracle_sl']} | {r['val_sl']} | {r['stat_sl']} | "
                 f"{r['aic_sl']} | {r['pacf_sl']} | {r['oracle_mse']:.4f} | "
                 f"{r['val_regret']:+.4f} | {r['stat_regret']:+.4f} | {r['aic_regret']:+.4f} | {r['pacf_regret']:+.4f} |")
lines.append('')
for name in ['val', 'stat', 'aic', 'pacf']:
    hits = sum(r[f'{name}_hit'] for r in rows)
    regs = [r[f'{name}_regret'] for r in rows]
    rels = [100 * r[f'{name}_regret'] / r['oracle_mse'] for r in rows]
    lines.append(f'{name:5s}: hit {hits}/16, mean regret {sum(regs)/16:.4f}, mean rel {sum(rels)/16:.2f}%, '
                 f'median rel {sorted(rels)[8]:.2f}%')
with open('logs/ehs_fix/e3_verdict.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
