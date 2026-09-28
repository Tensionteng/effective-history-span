#!/usr/bin/env python
"""Unit tests for InputDecayWrapper (data_provider/data_factory.py).

Acceptance: mu=1.0 is the identity, underlying data not polluted, shape unchanged;
plus weight-direction and auto-mu resolution (drift -> lambda -> mu) checks.

Run: .venv/bin/python analysis/decay_unittest.py
"""
import os
import sys

import numpy as np
from torch.utils.data import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_provider.data_factory import InputDecayWrapper, _drift_to_lambda, POWER_LAW_C

L, C = 2880, 4


class FakeBase(Dataset):
    """Mimics Dataset_Custom: returns numpy views into persistent arrays."""

    def __init__(self, data):
        self.data_x = data
        self.data_y = data
        self.seq_len = L
        self.pred_len = 96
        self.data_stamp = np.zeros((len(data), 2))

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, i):
        return (self.data_x[i:i + self.seq_len],
                self.data_y[i + self.seq_len - 48:i + self.seq_len + self.pred_len],
                self.data_stamp[i:i + self.seq_len],
                self.data_stamp[i + self.seq_len - 48:i + self.seq_len + self.pred_len])


def test_identity_mu1():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(L + 200, C))
    base = FakeBase(data)
    w = InputDecayWrapper(base, L, 1.0)
    for i in (0, 5, 100):
        out = w[i][0]
        assert np.array_equal(out, base[i][0]), 'mu=1.0 not identity'
    print('PASS identity_mu1')


def test_no_pollution_and_shape():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(L + 200, C))
    snap = data.copy()
    for mu in (0.995, 0.999, 1.0):
        w = InputDecayWrapper(FakeBase(data), L, mu)
        for i in range(0, len(w), 37):
            sx, sy, mx, my = w[i]
            assert sx.shape == (L, C), f'shape {sx.shape}'
            assert sy.shape == (48 + 96, C)
    assert np.array_equal(data, snap), 'underlying data polluted'
    print('PASS no_pollution_and_shape')


def test_weight_direction():
    data = np.ones((L + 200, C))
    base = FakeBase(data)
    mu = 0.999
    w = InputDecayWrapper(base, L, mu)
    out = w[0][0][:, 0]
    assert abs(out[-1] - 1.0) < 1e-12, f'recent step weight {out[-1]} != 1'
    assert abs(out[0] - mu ** (L - 1)) < 1e-9, f'oldest step weight {out[0]} != mu^(L-1)'
    assert np.all(np.diff(out) > 0), 'weights not monotone increasing toward recent'
    # seq_y untouched
    raw = base[0]
    assert np.array_equal(w[0][1], raw[1]), 'seq_y changed'
    print(f'PASS weight_direction (oldest={out[0]:.4f}, recent=1)')


def test_auto_mu_mapping():
    # drift above calibration range -> clamp to fastest lambda -> smallest N*
    lam_hi = _drift_to_lambda(0.9)
    lam_lo = _drift_to_lambda(0.005)
    lam_mid = _drift_to_lambda(0.09)
    assert lam_hi >= lam_mid >= lam_lo, 'lambda not monotone in drift'
    for drift, expect_clamped in ((0.9, True), (0.005, True), (0.09, False)):
        lam = _drift_to_lambda(drift)
        nstar = POWER_LAW_C * lam ** (-1.0 / 3.0)
        mu = np.exp(-1.0 / nstar)
        print(f'  drift={drift}: lambda={lam:.6f} N*={nstar:.0f} mu={mu:.6f} clamped={expect_clamped}')
    assert abs(_drift_to_lambda(0.9) - _drift_to_lambda(10.0)) < 1e-12, 'high end not clamped'
    print('PASS auto_mu_mapping')


def test_auto_mu_floor():
    """High-drift train data must hit the mu >= exp(-3/L) floor (N* >= L/3)."""
    from types import SimpleNamespace
    from data_provider.data_factory import _resolve_decay_auto_mu
    rng = np.random.default_rng(3)
    # random walk = extreme drift (drift_index >> calibration max)
    walk = np.cumsum(rng.normal(size=(4000, 2)), axis=0)
    base = FakeBase(walk)
    base.data_x = walk
    args = SimpleNamespace(seq_len=L, _decay_mu_resolved=None)
    del args._decay_mu_resolved  # exercise the getattr miss path
    mu = _resolve_decay_auto_mu(args, base, 'train')
    floor = np.exp(-3.0 / L)
    assert abs(mu - floor) < 1e-12, f'mu={mu:.8f} != floor {floor:.8f}'
    # cached: second call returns the same floored value without recompute
    assert _resolve_decay_auto_mu(args, base, 'train') == mu
    print(f'PASS auto_mu_floor (mu={mu:.6f} = floor, N*={-1 / np.log(mu):.0f} = L/3)')


if __name__ == '__main__':
    test_identity_mu1()
    test_no_pollution_and_shape()
    test_weight_direction()
    test_auto_mu_mapping()
    test_auto_mu_floor()
    print('ALL TESTS PASSED')
