"""EHS task C v2: cache per-channel Chronos errors at ALL candidate lookbacks,
then evaluate truncation rules offline (zero re-inference cost).

For each dataset, run bolt-base zero-shot at every L in {96, 336, 720, 1440}
on the identical test-target set, and cache per-window per-channel squared /
absolute errors to logs/ehs_final/taskC_cache_<ds>.npz. Rules then just pick a
column per channel and average.

Rules evaluated:
  fixed96/fixed336/fixed720/fixed1440 : all channels at L
  oracle    : per-channel argmin (upper bound of any per-series rule)
  quantile  : v1 rule, score quartiles -> {96,336,720,1440}
  thr       : absolute rule: drift>D_HI -> 96; elif cal>C_HI -> 1440;
              elif cal>C_MID -> 720; else 336
  thr+dataset: same but if dataset mean drift > D_DS, force ALL channels to 96
              (a dataset-level veto from the task-B finding that drift is the
              dominant dataset-level predictor)

Usage: CUDA_VISIBLE_DEVICES=5 .venv/bin/python tsfm/ehs_adaptive_v2.py \
    --datasets ETTh1,exchange_rate,weather,electricity [--test-stride 1] [--stride-electricity 2]
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour
from analysis.ehs_predict_v2 import channel_stats
from tsfm.ehs_adaptive import batched_predict, build_ds, DATASETS, CANDIDATES, PRED_LEN

CACHE = 'logs/ehs_final/taskC_cache_{}.npz'
OUT = 'logs/ehs_final/taskC_adaptive_v2.json'

# absolute-rule thresholds (on train-segment channel stats)
D_HI, C_HI, C_MID, D_DS = 0.70, 0.40, 0.12, 0.60


def cache_run(pipe, name, kind, root, path, periods, stride):
    """Run all 4 lookbacks, cache se/ae [nL, N, C] and channel stats."""
    test_ds = build_ds(kind, root, path, 'test', max(CANDIDATES))
    n = len(test_ds)
    idx = list(range(0, n, stride))
    sub = torch.utils.data.Subset(test_ds, idx)
    C = test_ds.data_x.shape[1]
    B = max(1, 2048 // C)
    loader = DataLoader(sub, batch_size=B, shuffle=False, num_workers=2, drop_last=False)

    train_ds = build_ds(kind, root, path, 'train', 96)
    feats = np.array([[channel_stats(train_ds.data_x[:, c], periods)[k]
                       for k in ('drift', 'calendar_acf_diff', 'long_acf_diff')]
                      for c in range(C)])

    se = np.zeros((len(CANDIDATES), len(idx), C), dtype=np.float32)
    ae = np.zeros_like(se)
    trues = None
    for li, L in enumerate(CANDIDATES):
        t0 = time.time()
        preds, trues = [], []
        for bx, by, _, _ in loader:
            bx = bx.float().cuda()
            Bb = bx.shape[0]
            flat = bx[:, -L:, :].permute(0, 2, 1).reshape(Bb * C, L)
            out = batched_predict(pipe, flat)
            p = out.reshape(Bb, C, PRED_LEN).transpose(0, 2, 1)
            t = by[:, -PRED_LEN:, :].float().numpy()
            preds.append(p)
            trues.append(t)
        p = np.concatenate(preds)
        t = np.concatenate(trues)
        se[li] = ((p - t) ** 2).mean(axis=1)  # [N, C]
        ae[li] = np.abs(p - t).mean(axis=1)
        print(f'[{name}] L={L}: mse={se[li].mean():.4f} ({time.time()-t0:.0f}s)', flush=True)
    np.savez_compressed(CACHE.format(name), se=se, ae=ae, feats=feats,
                        idx=np.array(idx), candidates=np.array(CANDIDATES))
    return se, ae, feats


def rule_lengths(rule, feats):
    """feats: [C, 3] (drift, calendar_acf_diff, long_acf_diff) -> per-channel L."""
    C = feats.shape[0]
    if rule == 'quantile':
        z = (feats - feats.mean(0)) / (feats.std(0) + 1e-12)
        score = z[:, 1] + z[:, 2] - z[:, 0]
        order = np.argsort(np.argsort(score))
        q = np.minimum(order * len(CANDIDATES) // C, len(CANDIDATES) - 1)
        return np.array([CANDIDATES[i] for i in q])
    if rule == 'thr':
        L = np.where(feats[:, 0] > D_HI, 96,
                     np.where(feats[:, 1] > C_HI, 1440,
                              np.where(feats[:, 1] > C_MID, 720, 336)))
        return L.astype(int)
    if rule == 'thr+dataset':
        if feats[:, 0].mean() > D_DS:
            return np.full(C, 96)
        return rule_lengths('thr', feats)
    raise ValueError(rule)


def evaluate(name, se, ae, feats):
    """se: [nL, N, C]. Returns per-rule mse/mae."""
    res = {}
    for li, L in enumerate(CANDIDATES):
        res[f'fixed{L}'] = {'mse': float(se[li].mean()), 'mae': float(ae[li].mean())}
    best_per_ch = se.argmin(axis=0)  # [N?] no: per channel -> use channel-mean se
    ch_se = se.mean(axis=1)          # [nL, C]
    ch_ae = ae.mean(axis=1)
    best = ch_se.argmin(axis=0)      # [C]
    res['oracle'] = {'mse': float(ch_se[best, np.arange(feats.shape[0])].mean()),
                     'mae': float(ch_ae[best, np.arange(feats.shape[0])].mean()),
                     'L_dist': {int(L): int((np.array(CANDIDATES)[best] == L).sum()) for L in CANDIDATES}}
    for rule in ('quantile', 'thr', 'thr+dataset'):
        Ls = rule_lengths(rule, feats)
        li = np.array([CANDIDATES.index(int(l)) for l in Ls])
        res[rule] = {'mse': float(ch_se[li, np.arange(feats.shape[0])].mean()),
                     'mae': float(ch_ae[li, np.arange(feats.shape[0])].mean()),
                     'L_dist': {int(L): int((Ls == L).sum()) for L in CANDIDATES}}
    best_fixed = min(res['fixed96']['mse'], res['fixed1440']['mse'])
    for rule in ('quantile', 'thr', 'thr+dataset', 'oracle'):
        res[rule]['le_best_fixed'] = bool(res[rule]['mse'] <= best_fixed + 1e-12)
    res['best_fixed_mse'] = best_fixed
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='ETTh1,exchange_rate,weather,electricity')
    ap.add_argument('--test-stride', type=int, default=1)
    ap.add_argument('--stride-electricity', type=int, default=2)
    ap.add_argument('--stride-traffic', type=int, default=2)
    args = ap.parse_args()

    os.makedirs('logs/ehs_final', exist_ok=True)
    results = {}
    if os.path.exists(OUT):
        results = json.load(open(OUT))

    pipe = None
    for name in args.datasets.split(','):
        kind, root, path, periods = DATASETS[name]
        stride = (args.stride_electricity if name == 'electricity'
                  else args.stride_traffic if name == 'traffic' else args.test_stride)
        cache = CACHE.format(name)
        if os.path.exists(cache):
            z = np.load(cache)
            se, ae, feats = z['se'], z['ae'], z['feats']
            print(f'[{name}] loaded cache {cache}')
        else:
            if pipe is None:
                from chronos import BaseChronosPipeline
                pipe = BaseChronosPipeline.from_pretrained(
                    'amazon/chronos-bolt-base', device_map='cuda', torch_dtype=torch.bfloat16)
            se, ae, feats = cache_run(pipe, name, kind, root, path, periods, stride)
        results[name] = evaluate(name, se, ae, feats)
        results[name]['test_stride'] = stride
        results[name]['n_windows'] = int(se.shape[1])
        results[name]['n_channels'] = int(se.shape[2])
        bf = results[name]['best_fixed_mse']
        line = '  '.join(f"{r}={results[name][r]['mse']:.4f}{'*' if results[name][r].get('le_best_fixed') else ''}"
                         for r in ('fixed96', 'fixed336', 'fixed720', 'fixed1440',
                                   'quantile', 'thr', 'thr+dataset', 'oracle'))
        print(f'[{name}] best_fixed={bf:.4f} | {line}', flush=True)
        with open(OUT, 'w') as f:
            json.dump(results, f, indent=1)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
