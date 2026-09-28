#!/usr/bin/env python
"""Merge E1 shards and build the Chronos-Bolt wide-data zero-shot table.

electricity L in {96,336,720,1440}: reused from logs/ehs_final/taskC_cache_electricity.npz
(same protocol: stride=2, same target set; deterministic zero-shot).
electricity L=2880 and all traffic arms: logs/ehs_fix/e1_<ds>_shard*of*.npz.

NOTE: chronos-bolt-base context_length=2048 -> the L=2880 arm is truncated to
the last 2048 steps inside the model (verified in code and numerically).

Outputs logs/ehs_fix/e1_merged.json + prints the table.
"""
import glob
import json
import re

import numpy as np

OUT = 'logs/ehs_fix/e1_merged.json'
LS = [96, 336, 720, 1440, 2880]


def merge_shards(name, nshards):
    """Return (Ls, se [nL, Ntot, C], ae, n_parts) assembled from shards."""
    parts = []
    for f in sorted(glob.glob(f'logs/ehs_fix/e1_{name}_shard*of{nshards}.npz')):
        z = np.load(f)
        parts.append((z['idx'], z['se'], z['ae'], [int(x) for x in z['Ls']]))
    if not parts:
        return None
    Ls = parts[0][3]
    se = np.concatenate([p[1] for p in parts], axis=1)
    ae = np.concatenate([p[2] for p in parts], axis=1)
    return Ls, se, ae, len(parts)


def main():
    out = {}

    # electricity: reuse taskC cache for L<=1440 (stride=2), add L=2880 from shards
    z = np.load('logs/ehs_final/taskC_cache_electricity.npz')
    se_old = z['se'].astype(np.float64)  # [4, N, C] for L in 96..1440
    ae_old = z['ae'].astype(np.float64)
    cands = [int(c) for c in z['candidates']]
    m = merge_shards('electricity', 2)
    assert m is not None, 'electricity L=2880 shards missing'
    Ls_new, se_new, ae_new, nsh = m
    assert Ls_new == [2880], Ls_new
    assert se_new.shape[1] == se_old.shape[1], (se_new.shape, se_old.shape)
    se = np.concatenate([se_old, se_new], axis=0)
    ae = np.concatenate([ae_old, ae_new], axis=0)
    Ls_e = cands + [2880]
    out['electricity'] = {
        'Ls': Ls_e,
        'mse': {str(L): float(se[i].mean()) for i, L in enumerate(Ls_e)},
        'mae': {str(L): float(ae[i].mean()) for i, L in enumerate(Ls_e)},
        'n_windows': int(se.shape[1]), 'channels': int(se.shape[2]),
        'stride': 2, 'source': 'taskC cache (L<=1440) + e1 shards (L=2880)',
        'per_channel_se': se.tolist(),
    }

    # traffic: all from shards
    m = merge_shards('traffic', 6)
    assert m is not None, 'traffic shards missing'
    Ls_t, se_t, ae_t, nsh = m
    assert Ls_t == LS, Ls_t
    out['traffic'] = {
        'Ls': Ls_t,
        'mse': {str(L): float(se_t[i].mean()) for i, L in enumerate(Ls_t)},
        'mae': {str(L): float(ae_t[i].mean()) for i, L in enumerate(Ls_t)},
        'n_windows': int(se_t.shape[1]), 'channels': int(se_t.shape[2]),
        'stride': 2, 'source': f'e1 shards ({nsh})',
        'per_channel_se': se_t.tolist(),
    }

    with open(OUT, 'w') as f:
        json.dump(out, f)

    for name, r in out.items():
        row = ' | '.join(f"L={L}: {r['mse'][str(L)]:.4f}" for L in r['Ls'])
        best = min(r['Ls'], key=lambda L: r['mse'][str(L)])
        print(f"{name} (N={r['n_windows']}, C={r['channels']}, stride={r['stride']}): {row} | best L={best}")


if __name__ == '__main__':
    main()
