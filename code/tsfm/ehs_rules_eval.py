"""Offline rule evaluation on cached per-channel Chronos errors (task C).

Reads logs/ehs_final/taskC_cache_<ds>.npz (se/ae [nL,N,C], feats [C,3]) and
evaluates rules without any re-inference:

  fixed{L}      all channels at L
  oracle        per-channel argmin (USES TEST ERRORS -> upper bound only)
  quantile      score-quartile rule (v1)
  thr           per-channel absolute thresholds
  thr+dataset   thr + dataset-level drift veto
  def1440+veto  default 1440 for every channel; if dataset mean drift > D_DS
                force all channels to 96 (dataset-level split only)

Writes logs/ehs_final/taskC_rules_final.json.
"""
import glob
import json
import os

import numpy as np

CANDIDATES = [96, 336, 720, 1440]
D_HI, C_HI, C_MID, D_DS = 0.70, 0.40, 0.12, 0.60
OUT = 'logs/ehs_final/taskC_rules_final.json'


def rule_sel(rule, feats):
    C = feats.shape[0]
    if rule == 'quantile':
        z = (feats - feats.mean(0)) / (feats.std(0) + 1e-12)
        score = z[:, 1] + z[:, 2] - z[:, 0]
        order = np.argsort(np.argsort(score))
        return np.minimum(order * len(CANDIDATES) // C, len(CANDIDATES) - 1)
    if rule == 'thr':
        L = np.where(feats[:, 0] > D_HI, 0, np.where(feats[:, 1] > C_HI, 3,
                                                     np.where(feats[:, 1] > C_MID, 2, 1)))
        return L.astype(int)
    if rule == 'thr+dataset':
        return np.zeros(C, dtype=int) if feats[:, 0].mean() > D_DS else rule_sel('thr', feats)
    if rule == 'def1440+veto':
        return np.zeros(C, dtype=int) if feats[:, 0].mean() > D_DS else np.full(C, 3)
    raise ValueError(rule)


def main():
    results = {}
    for cache in sorted(glob.glob('logs/ehs_final/taskC_cache_*.npz')):
        name = os.path.basename(cache)[len('taskC_cache_'):-4]
        z = np.load(cache)
        se, ae, feats = z['se'].mean(axis=1), z['ae'].mean(axis=1), z['feats']  # [nL,C]
        C = se.shape[1]
        res = {}
        for li, L in enumerate(CANDIDATES):
            res[f'fixed{L}'] = {'mse': float(se[li].mean()), 'mae': float(ae[li].mean())}
        best = se.argmin(axis=0)
        res['oracle(test-info, upper bound)'] = {
            'mse': float(se[best, np.arange(C)].mean()),
            'mae': float(ae[best, np.arange(C)].mean()),
            'L_dist': {int(L): int((np.array(CANDIDATES)[best] == L).sum()) for L in CANDIDATES}}
        bf = min(res['fixed96']['mse'], res['fixed1440']['mse'])
        res['best_fixed_mse'] = bf
        for rule in ('quantile', 'thr', 'thr+dataset', 'def1440+veto'):
            sel = rule_sel(rule, feats)
            m = float(se[sel, np.arange(C)].mean())
            res[rule] = {'mse': m,
                         'mae': float(ae[sel, np.arange(C)].mean()),
                         'le_best_fixed': bool(m <= bf + 1e-12),
                         'L_dist': {int(L): int((np.array(CANDIDATES)[sel] == i).sum())
                                    for i, L in enumerate(CANDIDATES)}}
        res['oracle_gap_%'] = 100 * (bf - res['oracle(test-info, upper bound)']['mse']) / bf
        results[name] = res
        line = '  '.join(f"{r}={res[r]['mse']:.4f}{' PASS' if res[r].get('le_best_fixed') else ''}"
                         for r in ('fixed96', 'fixed336', 'fixed720', 'fixed1440',
                                   'quantile', 'thr', 'thr+dataset', 'def1440+veto',
                                   'oracle(test-info, upper bound)'))
        print(f'[{name}] best_fixed={bf:.4f} oracle_gap={res["oracle_gap_%"]:.2f}%\n    {line}')
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
