#!/usr/bin/env python
"""Chronos-Bolt zero-shot L-scan on the GIFT-Eval task cache (GIFT-Eval protocol).

Per task: test = last windows*pred_len steps, rolling windows (distance=pred_len),
Term.SHORT prediction length (freq-dependent, from the official data.py map).
For every lookback L in {96,336,720,1440,2880} the context is the last
min(L, 2048, available) steps of the window input (bolt context_length=2048
truncates from the left anyway -- the L=2880 arm measures effective 2048,
disclosed). NaN inputs are passed through: bolt masks them natively.

Outputs per task to logs/gift_ext/scan/<task>.npz:
  quants  [nL, nS, W, 9, P]  bolt quantile forecasts (levels 0.1..0.9)
  target  [nS, W, P]
  snaive  [nS, W, P]         seasonal-naive forecast (calendar season)
  ctx_mase_scale [nS]        mean |diff_1| of the full train context (MASE denominator)
plus appends a row to logs/gift_ext/scan_summary.csv with per-L metrics:
  crps   = mean_weighted_quantile_loss over quantiles 0.1..0.9 (GIFT-Eval CRPS)
  mase   = gluonts-style MASE, seasonal_period=1 (GIFT-Eval MASE)
  mse / mae = on quantile-mean point forecast
  nmse   = MSE / seasonal-naive MSE on identical windows (seasonal-naive-norm MSE)

Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python logs/gift_ext/scan.py --tasks ett1_H,...
"""
import argparse
import csv
import json
import os
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import pandas as pd
import torch

LS = [96, 336, 720, 1440, 2880]
CTX_MAX = 2048
CACHE = 'logs/gift_ext/data/cache'
OUT_DIR = 'logs/gift_ext/scan'
SUMMARY = 'logs/gift_ext/scan_summary.csv'
QLEVELS = np.arange(0.1, 1.0, 0.1).astype(np.float32)  # 9 quantiles, bolt order


def load_task(task):
    df = pd.read_parquet(os.path.join(CACHE, f'{task}.parquet'))
    return [(r.series_id, np.asarray(r.target, dtype=np.float32)) for r in df.itertuples()]


def build_eval_set(series, P, W):
    """Returns contexts (list of np arrays), targets [nS, W, P]."""
    ctxs, tgts = [], []
    for _, v in series:
        n = len(v)
        t_list = []
        for w in range(W):
            end = n - (W - w) * P
            ctxs.append(v[:end])
            t_list.append(v[end:end + P])
        tgts.append(np.stack(t_list))
    return ctxs, np.stack(tgts).astype(np.float32)  # [nS, W, P]


def ffill_bfill(x):
    s = pd.Series(x).ffill().bfill()
    return s.to_numpy(dtype=np.float32)


def seasonal_naive(ctx, P, m):
    n = len(ctx)
    if m <= 1 or m > n:
        last = ctx[-1] if n else np.nan
        return np.full(P, last, dtype=np.float32)
    c = ffill_bfill(ctx)
    pat = c[n - m:]
    reps = int(np.ceil(P / m))
    return np.tile(pat, reps)[:P].astype(np.float32)


def mase_scale(ctx):
    c = ctx[~np.isnan(ctx)]
    if len(c) < 2:
        return np.nan
    d = np.abs(np.diff(c)).mean()
    return float(d)


def predict_all(pipe, ctxs, L, P):
    """ctxs: list of np arrays (ragged). Returns quants [M, 9, P] float32."""
    Leff = min(L, CTX_MAX)
    M = len(ctxs)
    flat = np.full((M, Leff), np.nan, dtype=np.float32)
    for i, c in enumerate(ctxs):
        take = c[-Leff:] if len(c) >= Leff else c
        flat[i, -len(take):] = take
    chunk = int(np.clip(3500 / (0.6 * (Leff // 16 + 2)), 64, 4096))
    outs = []
    with torch.no_grad():
        for i in range(0, M, chunk):
            t = torch.from_numpy(flat[i:i + chunk]).cuda()
            o = pipe.predict(t, prediction_length=P)  # [b, 9, P]
            outs.append(o.float().cpu().numpy())
    return np.concatenate(outs, 0)


def metrics_block(quants, tgts, snaives, scales):
    """quants [M,9,P]; tgts/snaives [M,P]; scales [M].
    NaN target timesteps are excluded (GIFT *_with_missing 口径: evaluate on
    observed values only); items with no valid step are dropped per-metric."""
    y = tgts.astype(np.float64)
    q = quants.astype(np.float64)
    vt = np.isfinite(y) & np.isfinite(q[:, 4]) & np.isfinite(q).all(axis=1)
    n_vt = vt.sum(axis=1)
    has = n_vt > 0
    yv = np.where(vt, y, 0.0)
    point = q.mean(axis=1)
    med = q[:, 4]
    # CRPS: pinball loss |y-q| * (tau if y>=q else 1-tau), GIFT aggregation:
    # per item 2*sum_tau,t QL / sum_t|y|, then mean over items
    pin = np.abs(y[:, None, :] - q) * np.where(y[:, None, :] >= q, QLEVELS[None, :, None],
                                               1 - QLEVELS[None, :, None])
    ql = np.where(vt[:, None, :], pin, 0.0)
    denom = np.abs(yv).sum(axis=1)
    ok = has & (denom > 1e-8)
    crps = float(np.mean((2 * ql.sum(axis=(1, 2)) / np.maximum(denom, 1e-8))[ok]))
    # MASE (seasonal_period=1, gluonts 口径): per item mean|err| / context |diff_1| mean
    ae = np.abs(np.where(vt, y - med, 0.0)).sum(axis=1) / np.maximum(n_vt, 1)
    okm = has & np.isfinite(scales) & (scales > 1e-8)
    mase = float(np.mean((ae / np.maximum(scales, 1e-8))[okm])) if okm.any() else np.nan
    # MSE / MAE / seasonal-naive-normalized MSE (all on valid steps)
    se_i = (np.where(vt, (point - y) ** 2, 0.0)).sum(axis=1) / np.maximum(n_vt, 1)
    mae_i = (np.abs(np.where(vt, y - point, 0.0))).sum(axis=1) / np.maximum(n_vt, 1)
    sn_se_i = (np.where(vt, (snaives.astype(np.float64) - y) ** 2, 0.0)).sum(axis=1) / np.maximum(n_vt, 1)
    ok2 = has & (sn_se_i > 1e-12)
    nmse = float(np.mean((se_i / np.maximum(sn_se_i, 1e-12))[ok2])) if ok2.any() else np.nan
    return {'crps': crps, 'mase': mase, 'mse': float(se_i[has].mean()),
            'mae': float(mae_i[has].mean()), 'nmse': nmse, 'n_items': int(has.sum()),
            'n_valid_crps': int(ok.sum()), 'n_valid_nmse': int(ok2.sum()),
            'nan_step_frac': float(1 - n_vt.sum() / vt.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default=None, help='comma list; default=all')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    meta = json.load(open('logs/gift_ext/tasks.json'))
    tasks = args.tasks.split(',') if args.tasks else list(meta)

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        'amazon/chronos-bolt-base', device_map='cuda', torch_dtype=torch.bfloat16)

    write_header = not os.path.exists(SUMMARY)
    with open(SUMMARY, 'a', newline='') as fsum:
        wr = csv.writer(fsum)
        if write_header:
            wr.writerow(['task', 'domain', 'freq', 'pred_len', 'windows', 'n_series',
                         'L', 'L_eff', 'crps', 'mase', 'mse', 'mae', 'nmse',
                         'n_items', 'elapsed_s'])
        for task in tasks:
            m = meta[task]
            P, W, season = m['pred_len'], m['windows'], m['season']
            t0 = time.time()
            series = load_task(task)
            ctxs, tgts = build_eval_set(series, P, W)
            nS = len(series)
            snaives = np.stack([[seasonal_naive(ctxs[s * W + w], P, season)
                                 for w in range(W)] for s in range(nS)])
            scales = np.array([mase_scale(v[:len(v) - W * P]) for _, v in series])
            quants = np.zeros((len(LS), nS, W, 9, P), dtype=np.float32)
            for li, L in enumerate(LS):
                tl = time.time()
                q = predict_all(pipe, ctxs, L, P)  # [nS*W, 9, P]
                quants[li] = q.reshape(nS, W, 9, P)
                mb = metrics_block(q, tgts.reshape(-1, P), snaives.reshape(-1, P),
                                   np.repeat(scales, W))
                wr.writerow([task, m['domain'], m['freq'], P, W, nS, L, min(L, CTX_MAX),
                             f"{mb['crps']:.6f}", f"{mb['mase']:.6f}", f"{mb['mse']:.6f}",
                             f"{mb['mae']:.6f}", f"{mb['nmse']:.6f}", mb['n_items'],
                             round(time.time() - tl, 1)])
                fsum.flush()
                print(f"[{task}] L={L}(eff {min(L,CTX_MAX)}): crps={mb['crps']:.4f} "
                      f"mase={mb['mase']:.4f} nmse={mb['nmse']:.4f} ({time.time()-tl:.0f}s)",
                      flush=True)
            np.savez_compressed(os.path.join(OUT_DIR, f'{task}.npz'),
                                quants=quants, target=tgts, snaive=snaives,
                                mase_scale=scales,
                                series_id=np.array([s for s, _ in series]),
                                Ls=np.array(LS))
            print(f'[{task}] DONE in {time.time()-t0:.0f}s -> scan/{task}.npz', flush=True)


if __name__ == '__main__':
    main()
