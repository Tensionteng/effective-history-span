"""On-the-fly synthetic time series corpus (kernel-synthesis style, no storage).

Each sample: random trend + 1-3 seasonal harmonics + AR(2) noise + optional level shifts,
then standardized. Deterministic per-index seeding for reproducibility.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class SynthTS(Dataset):
    def __init__(self, n_series=200000, length=512, seed=0):
        self.n = n_series
        self.length = length
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1000003 + i)
        L = self.length
        t = np.arange(L, dtype=np.float64) / L

        # trend: random degree polynomial
        deg = rng.integers(0, 3)
        x = np.zeros(L)
        for d in range(deg + 1):
            x += rng.normal(0, 1) * t ** d

        # seasonal harmonics
        for _ in range(rng.integers(1, 4)):
            period = rng.uniform(8, L / 2)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.2, 2.0)
            x += amp * np.sin(2 * np.pi * np.arange(L) / period + phase)

        # AR(2) noise
        a1, a2 = rng.uniform(-0.9, 0.9), rng.uniform(-0.5, 0.5)
        eps = rng.normal(0, rng.uniform(0.1, 1.0), L)
        ar = np.zeros(L)
        for k in range(2, L):
            ar[k] = a1 * ar[k - 1] + a2 * ar[k - 2] + eps[k]
        x += ar

        # level shifts
        for _ in range(rng.integers(0, 3)):
            x[rng.integers(L // 4, 3 * L // 4):] += rng.normal(0, 1.5)

        x = (x - x.mean()) / (x.std() + 1e-8)
        return torch.from_numpy(x.astype(np.float32))
