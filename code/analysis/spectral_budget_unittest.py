#!/usr/bin/env python
"""Unit tests for InputSpectralBudgetWrapper (data_provider/data_factory.py).

Covers the acceptance criteria: underlying data not polluted, window shape
unchanged, near segment untouched, out-of-band energy exactly zero (spectral),
plus semantic checks for level/zero/downsample variants and the data-driven
top-3 peak mode used for exchange_rate.

Run: .venv/bin/python analysis/spectral_budget_unittest.py
"""
import os
import sys

import numpy as np
from torch.utils.data import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_provider.data_factory import InputBudgetWrapper, InputSpectralBudgetWrapper

L, BUDGET, C = 2880, 1440, 6
LD = L - BUDGET  # far segment length


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


def make_signal(rng, n=L + 200, c=C, periods=(24, 168), extra_period=50, offset=3.0):
    t = np.arange(n)[:, None]
    x = offset * np.ones((n, c))
    for i, p in enumerate(periods):
        x = x + (1.5 + i) * np.sin(2 * np.pi * t / p + i)
    if extra_period:  # out-of-band component that must be removed by 'spectral'
        x = x + 2.0 * np.sin(2 * np.pi * t / extra_period)
    x = x + rng.normal(scale=0.05, size=(n, c))
    return x


def band_bins(n, periods, rel=0.05):
    k = np.arange(n // 2 + 1, dtype=np.float64)
    keep = np.zeros(n // 2 + 1, dtype=bool)
    for p in periods:
        for mult in (1, 2):
            k0 = mult * n / p
            hw = max(1.0, rel * k0)
            keep |= np.abs(k - k0) <= hw
    keep[0] = False
    return keep


def test_no_pollution_and_shape():
    rng = np.random.default_rng(0)
    data = make_signal(rng)
    snap = data.copy()
    for variant in ('zero', 'level', 'spectral', 'downsample'):
        base = FakeBase(data)
        w = InputSpectralBudgetWrapper(base, L, BUDGET, variant, 'electricity.csv')
        for i in range(0, len(w), 50):
            sx, sy, mx, my = w[i]
            assert sx.shape == (L, C), f'{variant}: shape {sx.shape}'
            assert sy.shape == (48 + 96, C)
        assert np.array_equal(data, snap), f'{variant}: underlying data polluted'
    print('PASS no_pollution_and_shape')


def test_recency_untouched():
    rng = np.random.default_rng(1)
    data = make_signal(rng)
    base = FakeBase(data)
    raw = base[7][0]
    for variant in ('zero', 'level', 'spectral', 'downsample'):
        w = InputSpectralBudgetWrapper(FakeBase(data), L, BUDGET, variant, 'electricity.csv')
        out = w[7][0]
        assert np.array_equal(out[L - BUDGET:], raw[L - BUDGET:]), f'{variant}: near segment changed'
    print('PASS recency_untouched')


def test_spectral_band_energy():
    rng = np.random.default_rng(2)
    data = make_signal(rng)
    base = FakeBase(data)
    w = InputSpectralBudgetWrapper(FakeBase(data), L, BUDGET, 'spectral', 'electricity.csv')
    out = w[3][0][:LD]
    F = np.fft.rfft(out - out.mean(axis=0, keepdims=True), axis=0)
    mag = np.abs(F)
    keep = band_bins(LD, (24, 168))
    total = (mag ** 2).sum()
    inband = (mag[keep] ** 2).sum()
    outband = (mag[~keep] ** 2).sum()
    assert outband / total < 1e-8, f'out-of-band energy ratio {outband / total}'
    assert inband / total > 0.99, f'in-band ratio {inband / total}'  # signal mostly periodic
    assert abs(out.mean()) < 1e-6, 'DC not removed'
    # the injected out-of-band period-50 component must be gone
    k50 = LD / 50
    assert mag[int(round(k50))].max() < 1e-6 * np.sqrt(total), 'period-50 component survived'
    # raw window has most energy OUTSIDE the kept bands (offset + period-50 + leakage)
    Fraw = np.abs(np.fft.rfft(base[3][0][:LD] - base[3][0][:LD].mean(axis=0), axis=0)) ** 2
    assert (Fraw[keep].sum() / Fraw.sum()) < 0.9, 'test signal lacks out-of-band energy'
    print('PASS spectral_band_energy (out-of-band ratio < 1e-8)')


def test_level_zero_downsample():
    rng = np.random.default_rng(3)
    data = make_signal(rng)
    base = FakeBase(data)
    raw = base[5][0]
    D = raw[:LD]
    wl = InputSpectralBudgetWrapper(FakeBase(data), L, BUDGET, 'level', 'electricity.csv')[5][0]
    assert np.allclose(wl[:LD], D - D.mean(axis=0, keepdims=True)), 'level != demeaned D'
    wz = InputSpectralBudgetWrapper(FakeBase(data), L, BUDGET, 'zero', 'electricity.csv')[5][0]
    wz_ref = InputBudgetWrapper(FakeBase(data), L, BUDGET)[5][0]
    assert np.array_equal(wz, wz_ref), 'zero != InputBudgetWrapper output'
    wd = InputSpectralBudgetWrapper(FakeBase(data), L, BUDGET, 'downsample', 'electricity.csv')[5][0]
    blocks = wd[:LD].reshape(LD // 4, 4, C)
    assert (blocks.std(axis=1) == 0).all(), 'downsample not block-constant'
    ref_means = D[:LD // 4 * 4].reshape(LD // 4, 4, C).mean(axis=1)
    assert np.allclose(blocks[:, 0, :], ref_means), 'downsample block means wrong'
    print('PASS level_zero_downsample')


def test_data_driven_peaks():
    rng = np.random.default_rng(4)
    true_periods = (300, 900, 60)
    data = make_signal(rng, periods=true_periods, extra_period=None, offset=2.0)
    w = InputSpectralBudgetWrapper(FakeBase(data), L, BUDGET, 'spectral', 'exchange_rate.csv')
    out = w[2][0][:LD]
    F = np.fft.rfft(out - out.mean(axis=0, keepdims=True), axis=0)
    mag2 = np.abs(F) ** 2
    nz = np.where(mag2.sum(axis=1) > 1e-12 * mag2.sum())[0]
    assert len(nz) <= 3 * (2 * max(1, round(0.05 * (LD / 60))) + 1), f'too many kept bins: {len(nz)}'
    # every true peak bin must sit inside a kept band
    kept = np.zeros(LD // 2 + 1, bool)
    kept[nz] = True
    for p in true_periods:
        k0 = LD / p
        hw = max(1.0, 0.05 * k0)
        band = np.abs(np.arange(LD // 2 + 1) - k0) <= hw
        assert (kept & band).any(), f'true peak period {p} (bin {k0:.1f}) not kept'
    # out-of-band energy is zero by mask construction (up to fft roundtrip roundoff)
    assert mag2[~kept].sum() / mag2.sum() < 1e-10, 'energy outside selected bands'
    assert abs(out.mean()) < 1e-6, 'DC not removed in data-driven mode'
    print('PASS data_driven_peaks (top-3, out-of-band energy == 0)')


def test_spectral_matches_calendar_mask():
    """Wrapper-internal mask equals the analytic band definition."""
    w = InputSpectralBudgetWrapper(FakeBase(np.zeros((L + 200, C))), L, BUDGET, 'spectral',
                                   'weather.csv')
    assert w.periods == (144, 1008)
    m = w._band_mask(LD)
    assert np.array_equal(m, band_bins(LD, (144, 1008)))
    w2 = InputSpectralBudgetWrapper(FakeBase(np.zeros((L + 200, C))), L, BUDGET, 'spectral',
                                    'exchange_rate.csv')
    assert w2.periods is None, 'exchange_rate must be data-driven'
    print('PASS spectral_matches_calendar_mask')


if __name__ == '__main__':
    test_no_pollution_and_shape()
    test_recency_untouched()
    test_spectral_band_energy()
    test_level_zero_downsample()
    test_data_driven_peaks()
    test_spectral_matches_calendar_mask()
    print('ALL TESTS PASSED')
