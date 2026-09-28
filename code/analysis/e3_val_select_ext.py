#!/usr/bin/env python
"""E3 extension: validation-based lookback selection vs statistic predictor vs test-oracle
on the 6 NEW datasets (logs/ehs_v2_ext, 180 logs; 6 datasets x {iTransformer, DLinear}).

Protocol identical to analysis/e3_val_select.py (old 8 datasets):
  - parse per-epoch "Vali Loss" min (early-stopping criterion) and final test mse
  - val-selected sl: argmin over sl of mean best-vali-loss across seeds
    (per-seed variant: each seed picks its own argmin, then average test mse)
  - oracle sl: argmin over sl of mean test mse
Statistic-predictor side: analysis/ehs_predict_v3.py regressor at n=28
  (ridge, features = top-k |spearman| inside fold, leave-one-dataset-out) --
  predicted tiers parsed from logs/ehs_v2_ext/taskB_predict_v3.txt (k=3 primary,
  k=2 sensitivity). LOO = the honest out-of-sample prediction for the new datasets.
Verdict question: does "statistic ties validation" (E3 verdict on the old 16 cells)
hold on the 12 new cells under the n=28 regressor?
Outputs logs/ehs_v2_ext/e3_ext_verdict.{json,md}.
"""
import json
import os
import re
from collections import defaultdict

LOGDIR = 'logs/ehs_v2_ext'
TASKB_TXT = 'logs/ehs_v2_ext/taskB_predict_v3.txt'
OLD_VAL = 'logs/ehs_fix/e3_val_select_trained.json'  # old 16 cells (val side, for pooled view)
OUT = 'logs/ehs_v2_ext/e3_ext_verdict.json'
MD = 'logs/ehs_v2_ext/e3_ext_verdict.md'
SEEDS = [2021, 2022, 2023]
SLS = [96, 336, 720, 1440, 2880]
NEW_DS = ['PEMS04', 'PEMS08', 'bikes', 'ercot', 'pedestrian', 'solar']

import sys
if '--taskb' in sys.argv:
    TASKB_TXT = sys.argv[sys.argv.index('--taskb') + 1]
if '--suffix' in sys.argv:
    _s = sys.argv[sys.argv.index('--suffix') + 1]
    OUT = OUT.replace('.json', f'{_s}.json')
    MD = MD.replace('.md', f'{_s}.md')

re_name = re.compile(r'^([A-Za-z0-9]+)_([A-Za-z]+)_sl(\d+)_s(\d+)\.log$')
re_vali = re.compile(r'Vali Loss: ([\d.]+)')
re_mse = re.compile(r'^mse:([\d.eE+-]+)', re.M)


def parse_log(path):
    with open(path, errors='ignore') as f:
        txt = f.read()
    valis = [float(v) for v in re_vali.findall(txt)]
    mses = re_mse.findall(txt)
    mse = float(mses[-1]) if mses else None
    return (min(valis) if valis else None), mse


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def parse_taskb_pred(k):
    """Predicted tiers from the k=<k> LOO block of taskB_predict_v3.txt."""
    txt = open(TASKB_TXT).read()
    block = txt.split(f'k={k}:')[1]
    pred = {}
    for m in re.finditer(r'^\s+(\S+)\s+(iTransformer|DLinear)\s+true=\s*(\d+)\s+pred_tier=\s*(\d+)',
                         block, re.M):
        pred[(m.group(1), m.group(2))] = int(m.group(4))
    return pred


def main():
    # data[dataset][model][sl][seed] = (vali, mse)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    n_logs = n_parsed = 0
    for fn in sorted(os.listdir(LOGDIR)):
        m = re_name.match(fn)
        if not m:
            continue
        n_logs += 1
        ds, model, sl, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        vali, mse = parse_log(os.path.join(LOGDIR, fn))
        if vali is None or mse is None:
            continue
        n_parsed += 1
        data[ds][model][sl][seed] = (vali, mse)
    assert sorted(data) == sorted(NEW_DS), f'unexpected datasets: {sorted(data)}'

    pred_k3 = parse_taskb_pred(3)
    pred_k2 = parse_taskb_pred(2)

    rows = []
    for ds in sorted(data):
        for model in sorted(data[ds]):
            arms = data[ds][model]
            mean_vali = {sl: mean([v[0] for v in seeds.values()]) for sl, seeds in arms.items()}
            mean_mse = {sl: mean([v[1] for v in seeds.values()]) for sl, seeds in arms.items()}
            val_sl = min(mean_vali, key=mean_vali.get)
            ora_sl = min(mean_mse, key=mean_mse.get)
            stat_sl = pred_k3[(ds, model)]
            stat_sl_k2 = pred_k2[(ds, model)]
            per_seed = []
            for seed in SEEDS:
                avail = {sl: arms[sl][seed] for sl in arms if seed in arms[sl]}
                if not avail:
                    continue
                v_sl = min(avail, key=lambda s: avail[s][0])
                o_sl = min(avail, key=lambda s: avail[s][1])
                per_seed.append({'seed': seed, 'val_sl': v_sl, 'oracle_sl': o_sl,
                                 'mse_val': avail[v_sl][1], 'mse_oracle': avail[o_sl][1],
                                 'regret': avail[v_sl][1] - avail[o_sl][1]})
            ps_mse_val = mean([p['mse_val'] for p in per_seed])
            ps_mse_ora = mean([p['mse_oracle'] for p in per_seed])
            row = {
                'dataset': ds, 'model': model,
                'val_sl': val_sl, 'oracle_sl': ora_sl,
                'stat_sl_k3': stat_sl, 'stat_sl_k2': stat_sl_k2,
                'mse_oracle': mean_mse[ora_sl],
                'mse_val_selected': mean_mse[val_sl],
                'mse_stat_k3': mean_mse[stat_sl], 'mse_stat_k2': mean_mse[stat_sl_k2],
                'val_regret': mean_mse[val_sl] - mean_mse[ora_sl],
                'stat_regret_k3': mean_mse[stat_sl] - mean_mse[ora_sl],
                'stat_regret_k2': mean_mse[stat_sl_k2] - mean_mse[ora_sl],
                'val_hit': val_sl == ora_sl,
                'stat_hit_k3': stat_sl == ora_sl, 'stat_hit_k2': stat_sl_k2 == ora_sl,
                'mean_vali': {str(k): v for k, v in sorted(mean_vali.items())},
                'mean_mse': {str(k): v for k, v in sorted(mean_mse.items())},
                'per_seed': per_seed,
                'per_seed_mse_val': ps_mse_val, 'per_seed_mse_oracle': ps_mse_ora,
                'per_seed_regret': (ps_mse_val - ps_mse_ora) if ps_mse_val is not None else None,
            }
            rows.append(row)

    def agg(rows, key, hit_key):
        regs = [r[key] for r in rows]
        rels = [100 * r[key] / r['mse_oracle'] for r in rows]
        hits = sum(r[hit_key] for r in rows)
        srels = sorted(rels)
        med = (srels[len(srels) // 2] if len(srels) % 2
               else 0.5 * (srels[len(srels) // 2 - 1] + srels[len(srels) // 2]))
        return {'hit': hits, 'n': len(rows), 'mean_regret': sum(regs) / len(regs),
                'mean_rel_pct': sum(rels) / len(rels), 'median_rel_pct': med}

    summary = {
        'n_logs': n_logs, 'n_parsed': n_parsed,
        'new12': {'val': agg(rows, 'val_regret', 'val_hit'),
                  'stat_k3': agg(rows, 'stat_regret_k3', 'stat_hit_k3'),
                  'stat_k2': agg(rows, 'stat_regret_k2', 'stat_hit_k2'),
                  'per_seed_val_mean_regret': mean([r['per_seed_regret'] for r in rows])},
    }

    # pooled view: old 16 cells + new 12. Old val side from
    # logs/ehs_fix/e3_val_select_trained.json (all 8 datasets; e3_val_select.py's
    # filename regex was fixed to keep exchange_rate); stat side re-derived under
    # the n=28 LOO regressor from the taskB prediction file.
    pooled = list(rows)
    old = json.load(open(OLD_VAL))
    old_cells = []

    def add_old_cell(ds, model, mean_mse, val_sl):
        ora_sl = min(mean_mse, key=mean_mse.get)
        stat_sl = pred_k3[(ds, model)]
        old_cells.append({'dataset': ds, 'model': model,
                          'oracle_sl': ora_sl, 'mse_oracle': mean_mse[ora_sl],
                          'val_sl': val_sl, 'stat_sl_k3': stat_sl,
                          'val_regret': mean_mse[val_sl] - mean_mse[ora_sl],
                          'stat_regret_k3': mean_mse[stat_sl] - mean_mse[ora_sl],
                          'val_hit': val_sl == ora_sl, 'stat_hit_k3': stat_sl == ora_sl})

    for r in old:
        if r['model'] not in ('iTransformer', 'DLinear'):
            continue
        add_old_cell(r['dataset'], r['model'],
                     {int(k): v for k, v in r['mean_mse'].items()}, r['val_sl'])
    pooled28 = old_cells + [{k: r[k] for k in ('dataset', 'model', 'oracle_sl', 'mse_oracle',
                                               'val_sl', 'stat_sl_k3', 'val_regret',
                                               'stat_regret_k3', 'val_hit', 'stat_hit_k3')}
                            for r in rows]
    summary['pooled28'] = {'val': agg(pooled28, 'val_regret', 'val_hit'),
                           'stat_k3': agg(pooled28, 'stat_regret_k3', 'stat_hit_k3')}

    with open(OUT, 'w') as f:
        json.dump({'rows': rows, 'old_cells_n28stat': old_cells, 'summary': summary}, f, indent=2)

    lines = [
        '# E3-ext: validation selector vs statistic predictor (n=28 LOO) vs oracle',
        '# 6 new datasets x {iTransformer, DLinear} = 12 cells; logs logs/ehs_v2_ext/*.log',
        f'# parsed {n_parsed}/{n_logs} logs (3 seeds x 5 sl per cell)',
        '',
        '| dataset | model | oracle sl | val sl | stat sl (k=3) | mse(oracle) | regret val | rel% | regret stat | rel% |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['oracle_sl']} | {r['val_sl']} | "
                     f"{r['stat_sl_k3']} | {r['mse_oracle']:.4f} | {r['val_regret']:+.4f} | "
                     f"{100 * r['val_regret'] / r['mse_oracle']:+.2f} | {r['stat_regret_k3']:+.4f} | "
                     f"{100 * r['stat_regret_k3'] / r['mse_oracle']:+.2f} |")
    s = summary['new12']
    lines += ['',
              f"new-12 | val     : hit {s['val']['hit']}/12, mean regret {s['val']['mean_regret']:+.4f}, "
              f"mean rel {s['val']['mean_rel_pct']:+.2f}%, median rel {s['val']['median_rel_pct']:+.2f}%",
              f"new-12 | stat k=3: hit {s['stat_k3']['hit']}/12, mean regret {s['stat_k3']['mean_regret']:+.4f}, "
              f"mean rel {s['stat_k3']['mean_rel_pct']:+.2f}%, median rel {s['stat_k3']['median_rel_pct']:+.2f}%",
              f"new-12 | stat k=2: hit {s['stat_k2']['hit']}/12, mean regret {s['stat_k2']['mean_regret']:+.4f}, "
              f"mean rel {s['stat_k2']['mean_rel_pct']:+.2f}%, median rel {s['stat_k2']['median_rel_pct']:+.2f}%",
              f"new-12 | per-seed val: mean regret {s['per_seed_val_mean_regret']:+.4f}",
              '',
              'pooled 28 cells (old = 16 from logs/ehs_fix/e3_val_select_trained.json '
              '(e3_val_select.py regex now keeps exchange_rate); stat side re-derived '
              'under the n=28 LOO regressor):']
    p = summary['pooled28']
    lines += [
        f"pooled-28 | val     : hit {p['val']['hit']}/28, mean regret {p['val']['mean_regret']:+.4f}, "
        f"mean rel {p['val']['mean_rel_pct']:+.2f}%, median rel {p['val']['median_rel_pct']:+.2f}%",
        f"pooled-28 | stat k=3: hit {p['stat_k3']['hit']}/28, mean regret {p['stat_k3']['mean_regret']:+.4f}, "
        f"mean rel {p['stat_k3']['mean_rel_pct']:+.2f}%, median rel {p['stat_k3']['median_rel_pct']:+.2f}%",
    ]
    with open(MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
