"""Synthetic time series with controlled regime shocks + ground-truth labels.

Each series (length 512): random trend + harmonics (one designated main harmonic)
+ AR(2) noise, then 0-3 shocks injected at random positions, 4 types:
  S0/S1: level step        (mean shifts up/down)
  S1/S2: volatility burst  (noise std steps up)
  S2/S3: trend change      (slope steps)
  S3/S4: frequency change  (main harmonic period steps, phase-continuous)

Shock magnitudes are scaled by the pre-shock series std so they stay salient
after whole-series standardization. Returns the shocked series, a no-shock
counterfactual (same base, same normalization), and ground-truth shocks.
Deterministic per-index seeding for reproducibility.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

LEVEL, VOL, TREND, FREQ = 0, 1, 2, 3
SHOCK_NAMES = ['level', 'vol', 'trend', 'freq']


def _gen_base(rng, L, noise_lo=0.2, noise_hi=0.8):
    """Trend + harmonics + AR(2) noise. Returns (x, meta) with meta used for shocks."""
    t = np.arange(L, dtype=np.float64) / L
    x = np.zeros(L)
    deg = rng.integers(0, 3)
    for d in range(deg + 1):
        x += rng.normal(0, 1) * t ** d

    # main harmonic (subject to S4 frequency shocks)
    main_period = rng.uniform(32, 128)
    main_amp = rng.uniform(0.5, 2.0)
    main_phase = rng.uniform(0, 2 * np.pi)
    x += main_amp * np.sin(2 * np.pi * np.arange(L) / main_period + main_phase)
    # extra static harmonics
    for _ in range(rng.integers(0, 3)):
        period = rng.uniform(8, L / 2)
        x += rng.uniform(0.2, 1.0) * np.sin(2 * np.pi * np.arange(L) / period + rng.uniform(0, 2 * np.pi))

    a1, a2 = rng.uniform(-0.9, 0.9), rng.uniform(-0.5, 0.5)
    while abs(a1 + a2) >= 0.99:  # keep stationary-ish
        a1, a2 = rng.uniform(-0.9, 0.9), rng.uniform(-0.5, 0.5)
    sigma0 = rng.uniform(noise_lo, noise_hi)
    eps = rng.normal(0, sigma0, L)
    ar = np.zeros(L)
    for k in range(2, L):
        ar[k] = a1 * ar[k - 1] + a2 * ar[k - 2] + eps[k]
    x += ar
    meta = dict(main_period=main_period, main_amp=main_amp, main_phase=main_phase,
                a1=a1, a2=a2, sigma0=sigma0, eps=eps)
    return x, meta


def _apply_shocks(base, meta, rng, L):
    """Apply 0-3 shocks to a copy of base. Returns (x, shocks list[(pos, type)])."""
    x = base.copy()
    std0 = base.std() + 1e-8
    n_shocks = rng.integers(0, 4)
    if n_shocks == 0:
        return x, []
    lo, hi = 48, L - 48
    pos = np.sort(rng.choice(np.arange(lo, hi), size=n_shocks, replace=False))
    # enforce a min gap of 32 steps (2 patches) so events are unambiguous
    for _ in range(64):
        if np.all(np.diff(pos) >= 32):
            break
        pos = np.sort(rng.choice(np.arange(lo, hi), size=n_shocks, replace=False))
    types = rng.choice(4, size=n_shocks, replace=False) if n_shocks <= 4 else None
    shocks = []
    for s, ty in zip(pos, types):
        s = int(s)
        ty = int(ty)
        if ty == LEVEL:
            x[s:] += rng.uniform(0.8, 2.0) * std0 * rng.choice([-1, 1])
        elif ty == VOL:
            factor = rng.uniform(2.5, 5.0)
            # re-run AR recursion from s with inflated noise std (same eps stream,
            # rescaled): approximate by adding extra AR-filtered noise
            extra = rng.normal(0, meta['sigma0'] * np.sqrt(factor ** 2 - 1.0), L - s)
            ar2 = np.zeros(L - s)
            for k in range(2, L - s):
                ar2[k] = meta['a1'] * ar2[k - 1] + meta['a2'] * ar2[k - 2] + extra[k]
            x[s:] += ar2
        elif ty == TREND:
            k = rng.uniform(0.8, 2.0) * std0 / 128.0 * rng.choice([-1, 1])
            x[s:] += k * (np.arange(L - s))
        elif ty == FREQ:
            factor = rng.uniform(1.5, 2.5) ** rng.choice([-1, 1])
            p1, p2 = meta['main_period'], meta['main_period'] * factor
            # phase-continuous piecewise-period main harmonic; subtract old, add new
            idx = np.arange(L, dtype=np.float64)
            phase = np.where(idx < s, idx / p1, s / p1 + (idx - s) / p2)
            old = meta['main_amp'] * np.sin(2 * np.pi * idx / p1 + meta['main_phase'])
            new = meta['main_amp'] * np.sin(2 * np.pi * phase + meta['main_phase'])
            x = x - old + new
        shocks.append((s, ty))
    return x, shocks


class ShockTS(Dataset):
    """Yields dict(x=[L] float32, base=[L] float32 no-shock, pos=[3] long, type=[3] long,
    n=n_shocks). pos/type padded with -1. Both x and base standardized with x's stats.

    noise_hi scales the AR innovation std range down for the low-noise (quiet) regime;
    shock magnitudes track the pre-shock std, so they stay salient relative to dynamics."""

    def __init__(self, n_series=20000, length=512, seed=0, noise_lo=0.2, noise_hi=0.8):
        self.n = n_series
        self.length = length
        self.seed = seed
        self.noise_lo = noise_lo
        self.noise_hi = noise_hi

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed * 1000003 + i) % (2 ** 63))
        L = self.length
        base, meta = _gen_base(rng, L, self.noise_lo, self.noise_hi)
        x, shocks = _apply_shocks(base, meta, rng, L)

        mu, sd = x.mean(), x.std() + 1e-8
        x = (x - mu) / sd
        base = (base - mu) / sd  # same transform -> counterfactual target

        pos = np.full(3, -1, dtype=np.int64)
        typ = np.full(3, -1, dtype=np.int64)
        for j, (s, ty) in enumerate(shocks[:3]):
            pos[j], typ[j] = s, ty
        return dict(x=torch.from_numpy(x.astype(np.float32)),
                    base=torch.from_numpy(base.astype(np.float32)),
                    pos=torch.from_numpy(pos), typ=torch.from_numpy(typ),
                    n=torch.tensor(len(shocks)))
