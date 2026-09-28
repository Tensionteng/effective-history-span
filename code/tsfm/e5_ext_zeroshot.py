#!/usr/bin/env python
"""E5-ext: Chronos-Bolt zero-shot on the 6 new EHS-v2-ext datasets.

Protocol identical to tsfm/e1_wide_zero_shot.py (which was identical to task C
v2, tsfm/ehs_adaptive_v2.py): test dataset built at seq_len=max(L) (TSL border
logic makes the test target set independent of seq_len), windows subsampled by
a per-dataset stride, channel-batched inference via batched_predict with
L-adaptive chunking.

NOTE: chronos-bolt-base has context_length=2048; inputs longer than 2048 are
truncated from the left inside ChronosBoltPipeline.predict (verified in the E1
run). The L=2880 arm therefore measures effective context 2048 -- disclosed in
the report.

Strides (protocol-internal, applied to all L arms of a dataset):
  ercot 1, pedestrian 1, PEMS04 2, PEMS08 2, solar 4, bikes 6.
Rationale: keep strided window counts in the 1.6k-2.6k band used by the E1
runs (electricity stride2 -> 2583 windows, traffic stride2 -> 1707) so that
per-dataset inference cost stays within a few GPU-hours; narrow sets
(ercot/pedestrian) run at stride 1.

Resumable: logs/ehs_v2_ext/zeroshot/zs_<ds>.npz stores se/ae [nL, N, C] with a
done mask; completed L arms are skipped on restart.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tsfm/e5_ext_zeroshot.py --dataset ercot
  .venv/bin/python tsfm/e5_ext_zeroshot.py --summarize
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsfm.ehs_adaptive import batched_predict, build_ds

LS = [96, 336, 720, 1440, 2880]
PRED_LEN = 96
OUT_DIR = 'logs/ehs_v2_ext/zeroshot'
# name: (kind, root, path, stride)
DS = {
    'ercot':      ('custom', './dataset/ercot/', 'ercot.csv', 1),
    'pedestrian': ('custom', './dataset/pedestrian/', 'pedestrian.csv', 1),
    'PEMS04':     ('custom', './dataset/PEMS04/', 'PEMS04.csv', 2),
    'PEMS08':     ('custom', './dataset/PEMS08/', 'PEMS08.csv', 2),
    'solar':      ('custom', './dataset/solar/', 'solar.csv', 4),
    'bikes':      ('custom', './dataset/bikes/', 'bikes.csv', 6),
}


def run_one(name, stride=None, Ls=LS, suffix=''):
    kind, root, path, default_stride = DS[name]
    stride = stride or default_stride
    cache = os.path.join(OUT_DIR, f'zs_{name}{suffix}.npz')

    test_ds = build_ds(kind, root, path, 'test', max(Ls))
    n = len(test_ds)
    idx = list(range(0, n, stride))
    sub = torch.utils.data.Subset(test_ds, idx)
    C = test_ds.data_x.shape[1]
    B = max(1, 2048 // C)
    loader = DataLoader(sub, batch_size=B, shuffle=False, num_workers=4, drop_last=False)
    print(f'[{name}] {len(idx)} windows (of {n}, stride={stride}), C={C}, batch={B}',
          flush=True)

    se = np.full((len(Ls), len(idx), C), np.nan, dtype=np.float32)
    ae = np.full_like(se, np.nan)
    done = np.zeros(len(Ls), dtype=bool)
    if os.path.exists(cache):
        z = np.load(cache)
        assert list(z['idx']) == idx and list(z['Ls']) == list(Ls), \
            f'cache {cache} does not match current idx/Ls'
        se, ae, done = z['se'], z['ae'], z['done']
        print(f'[{name}] resumed from {cache}: done={done.tolist()}', flush=True)
    if done.all():
        return

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        'amazon/chronos-bolt-base', device_map='cuda', torch_dtype=torch.bfloat16)
    print(f'model_context_length={pipe.model.config.chronos_config["context_length"]}',
          flush=True)

    for li, L in enumerate(Ls):
        if done[li]:
            continue
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
        done[li] = True
        print(f'[{name}] L={L}: mse={se[li].mean():.4f} mae={ae[li].mean():.4f} '
              f'({time.time() - t0:.0f}s)', flush=True)
        np.savez_compressed(cache, se=se, ae=ae, done=done,
                            idx=np.array(idx), Ls=np.array(Ls))


def summarize():
    """Merge zs_*.npz into zeroshot_summary.json (mse/mae per L, best L)."""
    out = {}
    for name, (_, _, _, stride) in DS.items():
        cache = os.path.join(OUT_DIR, f'zs_{name}.npz')
        if not os.path.exists(cache):
            print(f'[{name}] missing {cache}')
            continue
        z = np.load(cache)
        se, ae, done = z['se'], z['ae'], z['done']
        if not done.all():
            print(f'[{name}] incomplete: done={done.tolist()}')
            continue
        Ls = [int(x) for x in z['Ls']]
        mse = {str(L): float(se[li].mean()) for li, L in enumerate(Ls)}
        mae = {str(L): float(ae[li].mean()) for li, L in enumerate(Ls)}
        best_L = Ls[int(np.argmin([mse[str(L)] for L in Ls]))]
        # per-channel best-L distribution (channel-mean se argmin)
        ch = se.mean(axis=1)  # [nL, C]
        dist = {str(L): int((np.array(Ls)[ch.argmin(axis=0)] == L).sum()) for L in Ls}
        out[name] = {'Ls': Ls, 'mse': mse, 'mae': mae, 'best_L': best_L,
                     'per_channel_best_L': dist, 'n_windows': int(se.shape[1]),
                     'n_channels': int(se.shape[2]), 'stride': stride}
        print(f'[{name}] best_L={best_L} ' +
              ' '.join(f'L{L}={mse[str(L)]:.4f}' for L in Ls), flush=True)
    path = os.path.join(OUT_DIR, 'zeroshot_summary.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DS))
    ap.add_argument('--stride', type=int, default=None)
    ap.add_argument('--Ls', type=str, default=','.join(map(str, LS)))
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--suffix', default='',
                    help='cache file suffix, for running a subset of Ls into a side cache')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.summarize:
        summarize()
    else:
        run_one(args.dataset, args.stride, [int(x) for x in args.Ls.split(',')], args.suffix)


if __name__ == '__main__':
    main()
