#!/usr/bin/env python
"""E3 (part 1): validation-based lookback selection vs test-oracle, trained arms.

Parses logs/ehs_v2/{dataset}_{model}_sl{sl}_s{seed}.log for per-epoch Vali Loss
(early-stopping criterion) and the final test mse. Then, per (dataset, model):
  - val-selected sl: argmin over sl of mean best-vali-loss across seeds
    (per-seed variant: each seed picks its own argmin, then average test mse)
  - oracle sl:       argmin over sl of mean test mse
  - regret = test mse(val-selected) - test mse(oracle)
Outputs logs/ehs_fix/e3_val_select_trained.json + a markdown table.
"""
import json
import os
import re
from collections import defaultdict

LOGDIR = 'logs/ehs_v2'
OUT = 'logs/ehs_fix/e3_val_select_trained.json'
MD = 'logs/ehs_fix/e3_val_select_trained.md'
SEEDS = [2021, 2022, 2023]
SLS = [96, 336, 720, 1440, 2880]

re_name = re.compile(r'^(.+)_([A-Za-z]+)_sl(\d+)_s(\d+)\.log$')
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


def main():
    # data[dataset][model][sl][seed] = (vali, mse)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for fn in sorted(os.listdir(LOGDIR)):
        m = re_name.match(fn)
        if not m:
            continue
        ds, model, sl, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        vali, mse = parse_log(os.path.join(LOGDIR, fn))
        if vali is None or mse is None:
            continue
        data[ds][model][sl][seed] = (vali, mse)

    rows = []
    for ds in sorted(data):
        for model in sorted(data[ds]):
            arms = data[ds][model]
            # mean-based selection
            mean_vali = {sl: mean([v[0] for v in seeds.values()]) for sl, seeds in arms.items()}
            mean_mse = {sl: mean([v[1] for v in seeds.values()]) for sl, seeds in arms.items()}
            val_sl = min(mean_vali, key=mean_vali.get)
            ora_sl = min(mean_mse, key=mean_mse.get)
            # per-seed selection (each seed picks argmin vali over sl available for that seed)
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
            rows.append({
                'dataset': ds, 'model': model,
                'val_sl': val_sl, 'oracle_sl': ora_sl,
                'mse_val_selected': mean_mse[val_sl], 'mse_oracle': mean_mse[ora_sl],
                'regret': mean_mse[val_sl] - mean_mse[ora_sl],
                'rel_regret_pct': 100 * (mean_mse[val_sl] - mean_mse[ora_sl]) / mean_mse[ora_sl],
                'hit': val_sl == ora_sl,
                'mean_vali': {str(k): v for k, v in sorted(mean_vali.items())},
                'mean_mse': {str(k): v for k, v in sorted(mean_mse.items())},
                'per_seed': per_seed,
                'per_seed_mse_val': ps_mse_val, 'per_seed_mse_oracle': ps_mse_ora,
                'per_seed_regret': (ps_mse_val - ps_mse_ora) if ps_mse_val is not None else None,
            })

    os.makedirs('logs/ehs_fix', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(rows, f, indent=2)

    # markdown
    lines = ['| dataset | model | val-selected sl | oracle sl | hit | mse(val-sel) | mse(oracle) | regret | rel.% |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['val_sl']} | {r['oracle_sl']} | "
                     f"{'Y' if r['hit'] else 'n'} | {r['mse_val_selected']:.4f} | {r['mse_oracle']:.4f} | "
                     f"{r['regret']:+.4f} | {r['rel_regret_pct']:+.2f} |")
    hits = sum(r['hit'] for r in rows)
    regs = [r['regret'] for r in rows]
    rels = [r['rel_regret_pct'] for r in rows]
    ps_regs = [r['per_seed_regret'] for r in rows if r['per_seed_regret'] is not None]
    lines += ['',
              f'mean-based: exact-hit {hits}/{len(rows)}, mean regret {sum(regs)/len(regs):.4f}, '
              f'mean rel {sum(rels)/len(rels):.2f}%, median rel {sorted(rels)[len(rels)//2]:.2f}%',
              f'per-seed : mean regret {sum(ps_regs)/len(ps_regs):.4f} '
              f'(per-seed selection, n rows with per-seed data: {len(ps_regs)})']
    with open(MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
