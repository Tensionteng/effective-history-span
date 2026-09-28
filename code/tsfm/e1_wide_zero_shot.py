#!/usr/bin/env python
"""E1: Chronos-Bolt zero-shot on wide data (electricity/traffic) x L in {96,336,720,1440,2880}.

Protocol identical to task C v2 (tsfm/ehs_adaptive_v2.py): test dataset built at
seq_len=max(L) (TSL border logic makes the test target set independent of
seq_len; verified against SUMMARY tsfm table), windows subsampled by --stride
(electricity/traffic both use stride=2, noted in the report), channel-batched
inference via batched_predict with L-adaptive chunking.

NOTE: chronos-bolt-base has context_length=2048; inputs longer than 2048 are
truncated from the left inside ChronosBoltPipeline.predict (verified). The
L=2880 arm therefore measures effective context 2048 -- disclosed in the report.

Sharding: --shard i/n splits the strided window list into n contiguous slices
and runs slice i, writing logs/ehs_fix/e1_<ds>_shard{i}of{n}.npz. Merge with
--merge.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsfm.ehs_adaptive import batched_predict, build_ds

LS = [96, 336, 720, 1440, 2880]
PRED_LEN = 96
DS = {
    'electricity': ('custom', './dataset/electricity/', 'electricity.csv'),
    'traffic': ('custom', './dataset/traffic/', 'traffic.csv'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--Ls', type=str, default='96,336,720,1440,2880')
    args = ap.parse_args()

    kind, root, path = DS[args.dataset]
    Ls = [int(x) for x in args.Ls.split(',')]
    test_ds = build_ds(kind, root, path, 'test', max(Ls))
    n = len(test_ds)
    idx = list(range(0, n, args.stride))
    # contiguous shard of the strided index list
    bounds = np.linspace(0, len(idx), args.nshards + 1).astype(int)
    idx = idx[bounds[args.shard]:bounds[args.shard + 1]]
    sub = torch.utils.data.Subset(test_ds, idx)
    C = test_ds.data_x.shape[1]
    B = max(1, 2048 // C)
    loader = DataLoader(sub, batch_size=B, shuffle=False, num_workers=4, drop_last=False)
    print(f'[{args.dataset}] shard {args.shard}/{args.nshards}: {len(idx)} windows '
          f'(of {n}), C={C}, batch={B}', flush=True)

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        'amazon/chronos-bolt-base', device_map='cuda', torch_dtype=torch.bfloat16)
    print(f'model_context_length={pipe.model.config.chronos_config["context_length"]}', flush=True)

    se = np.zeros((len(Ls), len(idx), C), dtype=np.float32)
    ae = np.zeros_like(se)
    for li, L in enumerate(Ls):
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
        se[li] = ((p - t) ** 2).mean(axis=1)
        ae[li] = np.abs(p - t).mean(axis=1)
        print(f'[{args.dataset}] shard {args.shard} L={L}: mse={se[li].mean():.4f} '
              f'mae={ae[li].mean():.4f} ({time.time() - t0:.0f}s)', flush=True)
        np.savez_compressed(
            f'logs/ehs_fix/e1_{args.dataset}_shard{args.shard}of{args.nshards}.npz',
            se=se[:, :len(p)], ae=ae[:, :len(p)], idx=np.array(idx), Ls=np.array(Ls))


if __name__ == '__main__':
    main()
