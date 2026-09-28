"""EHS task C: Tier-1 adaptive truncation with Chronos bolt-base zero-shot.

Per-series rule: compute drift / calendar_acf_diff / long_acf_diff on the
standardized TRAIN segment of every channel (no test leakage), z-score them
within the dataset, score = z(calendar_acf_diff) + z(long_acf_diff) - z(drift),
map score quartiles to truncation lengths {96, 336, 720, 1440}.

Three arms evaluated on the IDENTICAL prediction-target set (Dataset with
seq_len=1440; the TSL border logic shifts border1 by seq_len so the test
targets are the same for every lookback):
  - fixed96:   every channel sees the last 96 steps
  - fixed1440: every channel sees the last 1440 steps
  - adaptive:  channel c sees the last L_c steps from the rule

Inference follows models/Chronos.py: flatten [B, L, C] -> [B*C, L], one
predict call (chunked), mean over quantiles/samples.

Usage:
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 .venv/bin/python tsfm/ehs_adaptive.py \
      --datasets ETTh1,exchange_rate,weather,electricity --test-stride 1
Outputs: logs/ehs_final/taskC_adaptive.json (+ stdout table)
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
from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
from analysis.ehs_predict_v2 import channel_stats

CANDIDATES = [96, 336, 720, 1440]
MAX_L = 1440
PRED_LEN = 96
DATASETS = {
    'ETTh1':        ('ett', './dataset/ETT-small/', 'ETTh1.csv', [24, 168]),
    'ETTh2':        ('ett', './dataset/ETT-small/', 'ETTh2.csv', [24, 168]),
    'ETTm1':        ('ettm', './dataset/ETT-small/', 'ETTm1.csv', [96, 672]),
    'ETTm2':        ('ettm', './dataset/ETT-small/', 'ETTm2.csv', [96, 672]),
    'exchange_rate': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', [5, 30]),
    'weather':      ('custom', './dataset/weather/', 'weather.csv', [144, 1008]),
    'electricity':  ('custom', './dataset/electricity/', 'electricity.csv', [24, 168]),
    'traffic':      ('custom', './dataset/traffic/', 'traffic.csv', [24, 168]),
}


def build_ds(kind, root, path, flag, seq_len):
    cls = Dataset_ETT_hour if kind == 'ett' else (Dataset_ETT_minute if kind == 'ettm' else Dataset_Custom)
    freq = 't' if kind == 'ettm' else 'h'
    return cls(SimpleNamespace(augmentation_ratio=0), root_path=root, flag=flag,
               size=[seq_len, 48, PRED_LEN], features='M', data_path=path,
               target='OT', scale=True, timeenc=1, freq=freq)


def pick_lengths(train_x, periods):
    """train_x: [T_train, C] standardized. Returns per-channel L in CANDIDATES."""
    C = train_x.shape[1]
    feats = np.array([[channel_stats(train_x[:, c], periods)[k]
                       for k in ('drift', 'calendar_acf_diff', 'long_acf_diff')]
                      for c in range(C)])  # [C, 3]
    z = (feats - np.nanmean(feats, 0)) / (np.nanstd(feats, 0) + 1e-12)
    score = z[:, 1] + z[:, 2] - z[:, 0]
    # quartile assignment: lowest 25% -> 96 ... highest 25% -> 1440
    order = np.argsort(np.argsort(score))  # rank 0..C-1
    q = np.minimum(order * len(CANDIDATES) // C, len(CANDIDATES) - 1)
    return np.array([CANDIDATES[i] for i in q]), score


def batched_predict(pipe, flat, chunk=None):
    """flat: [M, L] float32 cuda tensor -> [M, PRED_LEN] numpy.
    Chunk sized to keep bolt's intermediate activations within a few GiB."""
    L = flat.shape[1]
    if chunk is None:
        # bolt reserves ~0.6MB per patch-token per series (measured: 4.8MB @L=96,
        # 49MB @L=1440); keep each predict call within ~3.5 GiB so we can share
        # a GPU with other jobs.
        chunk = int(np.clip(3500 / (0.6 * (L // 16 + 2)), 64, 4096))
    outs = []
    for i in range(0, flat.shape[0], chunk):
        o = pipe.predict(flat[i:i + chunk], prediction_length=PRED_LEN)
        outs.append(o.mean(dim=1).float().cpu().numpy())
    return np.concatenate(outs, 0)


def run_arm(pipe, loader, Ls):
    """Ls: per-channel length (np.array [C]) or scalar int. Returns (preds, trues) lists."""
    preds, trues = [], []
    for bx, by, _, _ in loader:
        bx = bx.float().cuda()          # [B, MAX_L, C]
        B, _, C = bx.shape
        by = by[:, -PRED_LEN:, :].float().numpy()
        if np.isscalar(Ls):
            flat = bx[:, -Ls:, :].permute(0, 2, 1).reshape(B * C, -1)
            out = batched_predict(pipe, flat)
            pred = out.reshape(B, C, PRED_LEN).transpose(0, 2, 1)
        else:
            pred = np.zeros((B, PRED_LEN, C), dtype=np.float64)
            for L in CANDIDATES:
                idx = np.where(Ls == L)[0]
                if len(idx) == 0:
                    continue
                sub = bx[:, -L:, idx].permute(0, 2, 1).reshape(B * len(idx), L)
                out = batched_predict(pipe, sub)
                pred[:, :, idx] = out.reshape(B, len(idx), PRED_LEN).transpose(0, 2, 1)
        preds.append(pred)
        trues.append(by)
    return np.concatenate(preds), np.concatenate(trues)


def mse_mae(p, t):
    return float(np.mean((p - t) ** 2)), float(np.mean(np.abs(p - t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='ETTh1,exchange_rate,weather,electricity')
    ap.add_argument('--test-stride', type=int, default=1,
                    help='subsample test windows (protocol-internal, applies to ALL arms)')
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--out', default='logs/ehs_final/taskC_adaptive.json')
    args = ap.parse_args()

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        'amazon/chronos-bolt-base', device_map='cuda', torch_dtype=torch.bfloat16)

    results = {}
    for name in args.datasets.split(','):
        kind, root, path, periods = DATASETS[name]
        t0 = time.time()
        train_ds = build_ds(kind, root, path, 'train', 96)
        test_ds = build_ds(kind, root, path, 'test', MAX_L)
        n = len(test_ds)
        idx = list(range(0, n, args.test_stride))
        sub = torch.utils.data.Subset(test_ds, idx)
        C = test_ds.data_x.shape[1]
        B = max(1, 2048 // C)
        loader = DataLoader(sub, batch_size=B, shuffle=False, num_workers=2, drop_last=False)

        Ls, score = pick_lengths(train_ds.data_x, periods)
        dist = {int(L): int((Ls == L).sum()) for L in CANDIDATES}

        arms = {}
        for arm, spec in [('fixed96', 96), ('fixed1440', MAX_L), ('adaptive', Ls)]:
            p, t = run_arm(pipe, loader, spec)
            m, a = mse_mae(p, t)
            arms[arm] = {'mse': m, 'mae': a}
            print(f'[{name}] {arm}: mse={m:.4f} mae={a:.4f} ({time.time()-t0:.0f}s)', flush=True)
        best_fixed = min(arms['fixed96']['mse'], arms['fixed1440']['mse'])
        ok = arms['adaptive']['mse'] <= best_fixed + 1e-9
        results[name] = {'n_windows': len(idx), 'n_channels': int(C), 'L_dist': dist,
                         'arms': arms, 'best_fixed_mse': best_fixed, 'adaptive_le_best_fixed': bool(ok),
                         'periods': periods, 'test_stride': args.test_stride,
                         'elapsed_s': round(time.time() - t0, 1)}
        print(f'[{name}] adaptive={arms["adaptive"]["mse"]:.4f} vs best_fixed={best_fixed:.4f} -> '
              f'{"PASS" if ok else "FAIL"}; L_dist={dist}', flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
