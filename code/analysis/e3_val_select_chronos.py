#!/usr/bin/env python
"""E3 (part 2): Chronos zero-shot validation-selection from cached per-channel errors.

The taskC caches (logs/ehs_final/taskC_cache_<ds>.npz) hold per-window squared
errors se[nL, N, C] on the test-target set for L in {96,336,720,1440}. Chronos
is zero-shot (no training), so there is no canonical "validation" use; we slice
the cached test windows in temporal order: first 50% = selection set,
last 50% = evaluation set. This mimics val->test temporal ordering at zero
re-inference cost.

Arms (dataset-level): val-selected L (argmin mean SE on selection windows),
oracle L (argmin on evaluation windows), fixed96/1440, def1440+veto
(dataset mean drift > 0.60 -> 96 else 1440; the task-C statistic rule).
Per-channel variants (val-sel per channel, oracle per channel) are reported
as secondary.

Outputs logs/ehs_fix/e3_val_select_chronos.json + .md
"""
import json
import os

import numpy as np

CACHE = 'logs/ehs_final/taskC_cache_{}.npz'
OUT = 'logs/ehs_fix/e3_val_select_chronos.json'
MD = 'logs/ehs_fix/e3_val_select_chronos.md'
D_DS = 0.60  # task-C dataset-level drift veto threshold
DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather', 'electricity', 'traffic']


def main():
    rows = []
    for name in DATASETS:
        z = np.load(CACHE.format(name))
        se = z['se'].astype(np.float64)  # [nL, N, C]
        feats = z['feats']  # [C, 3] = drift, calendar_acf_diff, long_acf_diff
        cands = [int(c) for c in z['candidates']]
        nL, N, C = se.shape
        nsel = N // 2
        sel = se[:, :nsel, :].mean(axis=(1, 2))     # [nL] dataset-level sel error
        eva = se[:, nsel:, :].mean(axis=(1, 2))     # [nL] dataset-level eval error
        # per-channel selection/eval errors
        sel_pc = se[:, :nsel, :].mean(axis=1)       # [nL, C]
        eva_pc = se[:, nsel:, :].mean(axis=1)       # [nL, C]

        val_li = int(np.argmin(sel))
        ora_li = int(np.argmin(eva))
        drift_mean = float(feats[:, 0].mean())
        rule_li = cands.index(96) if drift_mean > D_DS else cands.index(1440)

        # per-channel selection
        val_pc_choice = np.argmin(sel_pc, axis=0)   # [C]
        ora_pc_choice = np.argmin(eva_pc, axis=0)
        mse_val_pc = eva_pc[val_pc_choice, np.arange(C)].mean()
        mse_ora_pc = eva_pc[ora_pc_choice, np.arange(C)].mean()

        rows.append({
            'dataset': name, 'n_sel': nsel, 'n_eval': N - nsel, 'channels': C,
            'drift_mean': drift_mean,
            'eval_mse_by_L': {str(cands[i]): float(eva[i]) for i in range(nL)},
            'val_sl': cands[val_li], 'oracle_sl': cands[ora_li],
            'mse_val_selected': float(eva[val_li]),
            'mse_oracle': float(eva[ora_li]),
            'regret': float(eva[val_li] - eva[ora_li]),
            'rel_regret_pct': float(100 * (eva[val_li] - eva[ora_li]) / eva[ora_li]),
            'hit': cands[val_li] == cands[ora_li],
            'rule_sl': cands[rule_li], 'mse_rule': float(eva[rule_li]),
            'rule_regret': float(eva[rule_li] - eva[ora_li]),
            'mse_fixed96': float(eva[cands.index(96)]),
            'mse_fixed1440': float(eva[cands.index(1440)]),
            'mse_val_perchannel': float(mse_val_pc),
            'mse_oracle_perchannel': float(mse_ora_pc),
        })

    with open(OUT, 'w') as f:
        json.dump(rows, f, indent=2)

    lines = [
        'selection = first 50% of cached test windows (temporal); evaluation = last 50%.',
        '',
        '| dataset | val-sel sl | oracle sl | hit | mse(val-sel) | mse(rule def1440+veto @sl) | mse(oracle) | regret val | regret rule |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['val_sl']} | {r['oracle_sl']} | {'Y' if r['hit'] else 'n'} | "
            f"{r['mse_val_selected']:.4f} | {r['mse_rule']:.4f} (@{r['rule_sl']}) | {r['mse_oracle']:.4f} | "
            f"{r['regret']:+.4f} | {r['rule_regret']:+.4f} |")
    lines.append('')
    lines.append('| dataset | mse fixed96 | mse fixed1440 | mse val-sel (per-channel) | mse oracle (per-channel) |')
    lines.append('|---|---|---|---|---|')
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['mse_fixed96']:.4f} | {r['mse_fixed1440']:.4f} | "
                     f"{r['mse_val_perchannel']:.4f} | {r['mse_oracle_perchannel']:.4f} |")
    with open(MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
