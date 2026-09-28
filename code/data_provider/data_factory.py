import json
import os
import numpy as np
import torch
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader, Dataset

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


class InputBudgetWrapper(Dataset):
    """Zero out input-window steps older than the prediction budget.

    seq_x[:-budget] is set to 0 (in the dataset's scaled units), keeping only
    the most recent `budget` steps of the encoder input; seq_y and time marks
    pass through unchanged. Applied to every split so training and inference
    see the same budgeted inputs. Activated by --input_budget > 0 (< seq_len).
    """

    def __init__(self, base, seq_len, budget):
        self.base = base
        self.budget = min(budget, seq_len)

    def __getattr__(self, name):  # delegate scaler/inverse_transform/etc.
        return getattr(self.base, name)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.base[index]
        seq_x = seq_x.copy()  # underlying arrays are views into data_x
        seq_x[:-self.budget] = 0.0
        return seq_x, seq_y, seq_x_mark, seq_y_mark


# calendar main periods (in time steps) kept by the 'spectral' variant, by csv stem;
# stems not listed here (e.g. exchange_rate) use data-driven per-channel top-3 peaks
SPECTRAL_CALENDAR_PERIODS = {
    'ETTh1': (24, 168), 'ETTh2': (24, 168),
    'ETTm1': (96, 672), 'ETTm2': (96, 672),
    'electricity': (24, 168), 'traffic': (24, 168),
    'weather': (144, 1008),
}
SPECTRAL_BAND_REL = 0.05  # ±5% frequency band around each kept periodicity
SPECTRAL_TOP_PEAKS = 3    # data-driven mode: peaks kept per channel
DOWNSAMPLE_FACTOR = 4     # 'downsample' variant: far segment at 1/4 resolution


class InputSpectralBudgetWrapper(Dataset):
    """Two-channel spectral history budget (EHS).

    Splits the encoder input window into a far segment D = seq_x[:seq_len-budget]
    and a near segment R = seq_x[seq_len-budget:], and rewrites D per `variant`:

      zero       D := 0 (hard input budget, same semantics as InputBudgetWrapper)
      level      D := D - mean(D) (segment level removed, dynamics kept as-is)
      spectral   D demeaned -> rfft over time -> keep only main-periodicity bands
                 (calendar periods ±5% plus their 2nd harmonic; data-driven
                 per-channel top-3 non-DC peaks ±5% when the dataset has no
                 calendar entry) -> irfft back to the original length
      downsample D := 4x block-mean downsample, repeat-upsampled back to full
                 length (SPRINT-lite: far history kept at 1/4 time resolution)

    R always passes through unchanged, so window shape, model and parameter
    count are untouched. Applied to every split so training and inference see
    the same transformed inputs. Activated by --spectral_budget != 'none'
    together with --input_budget b (0 < b < seq_len).
    """

    def __init__(self, base, seq_len, budget, variant, data_path=''):
        assert variant in ('zero', 'level', 'spectral', 'downsample')
        self.base = base
        self.budget = min(budget, seq_len)
        self.variant = variant
        stem = os.path.splitext(os.path.basename(data_path))[0]
        self.periods = SPECTRAL_CALENDAR_PERIODS.get(stem)  # None -> data-driven
        self._mask_cache = {}

    def __getattr__(self, name):  # delegate scaler/inverse_transform/etc.
        return getattr(self.base, name)

    def __len__(self):
        return len(self.base)

    def _band_mask(self, n):
        """Boolean keep-mask [n//2+1] of calendar bands for a length-n segment."""
        mask = self._mask_cache.get(n)
        if mask is None:
            k = np.arange(n // 2 + 1, dtype=np.float64)
            mask = np.zeros(n // 2 + 1, dtype=bool)
            for p in self.periods:
                for mult in (1, 2):  # fundamental and 2nd harmonic
                    k0 = mult * n / p
                    hw = max(1.0, SPECTRAL_BAND_REL * k0)
                    mask |= np.abs(k - k0) <= hw
            mask[0] = False  # DC stays excluded (segment is demeaned anyway)
            self._mask_cache[n] = mask
        return mask

    @staticmethod
    def _topk_peak_mask(mag, top=SPECTRAL_TOP_PEAKS):
        """Per-channel greedy top-`top` non-DC peaks, each widened ±5% (>=1 bin)."""
        K, C = mag.shape
        keep = np.zeros((K, C), dtype=bool)
        remaining = mag.copy()
        remaining[0] = 0.0  # DC excluded
        for _ in range(top):
            peak = remaining.argmax(axis=0)
            for c in range(C):
                k = peak[c]
                if remaining[k, c] <= 0:
                    continue
                hw = max(1, int(round(SPECTRAL_BAND_REL * k)))
                lo, hi = max(1, k - hw), min(K, k + hw + 1)
                keep[lo:hi, c] = True
                remaining[lo:hi, c] = 0.0
        return keep

    def _spectral(self, D):
        n = D.shape[0]
        Dm = D - D.mean(axis=0, keepdims=True)
        F = torch.fft.rfft(torch.from_numpy(Dm), dim=0)
        if self.periods is None:
            keep = self._topk_peak_mask(F.abs().numpy())
        else:
            keep = self._band_mask(n)[:, None]  # [K,1] broadcasts over channels
        F = F * torch.from_numpy(keep).to(F.dtype)
        return torch.fft.irfft(F, n=n, dim=0).numpy()

    @staticmethod
    def _downsample(D):
        n = D.shape[0]
        k = n // DOWNSAMPLE_FACTOR
        out = np.repeat(D[:k * DOWNSAMPLE_FACTOR].reshape(k, DOWNSAMPLE_FACTOR, -1).mean(axis=1),
                        DOWNSAMPLE_FACTOR, axis=0)
        if k * DOWNSAMPLE_FACTOR < n:  # hold the last block mean over the tail
            out = np.concatenate([out, np.repeat(out[-1:], n - k * DOWNSAMPLE_FACTOR, axis=0)], axis=0)
        return out

    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.base[index]
        seq_x = seq_x.copy()  # underlying arrays are views into data_x
        cut = seq_x.shape[0] - self.budget
        if self.variant == 'zero':
            seq_x[:cut] = 0.0
        elif self.variant == 'level':
            seq_x[:cut] -= seq_x[:cut].mean(axis=0, keepdims=True)
        elif self.variant == 'spectral':
            seq_x[:cut] = self._spectral(seq_x[:cut])
        elif self.variant == 'downsample':
            seq_x[:cut] = self._downsample(seq_x[:cut])
        return seq_x, seq_y, seq_x_mark, seq_y_mark


# --- drift-matched exponential forgetting (EHS) ---
# power-law relation measured in logs/power_law: W* = c * lam^(-1/3), c = exp(intercept)
# refit with slope fixed at -1/3 over the 6-level lambda ladder (grid-argmin W*)
POWER_LAW_C = 65.63
POWER_LAW_CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                               'logs', 'power_law', 'calibration.json')


def _drift_index(x, n_seg=20):  # same statistic as analysis/ehs_stats.py
    segs = np.array_split(x, n_seg)
    means = np.array([s.mean() for s in segs if len(s)])
    return float(means.var() / (x.var() + 1e-12))


def _train_split_drift(args, data_set, flag):
    """Drift index of the TRAIN split, ehs_stats.dataset_stats conventions
    (n_seg=20, first 12000 steps, up to 50 channels, seed 0)."""
    if flag == 'train':
        X = data_set.data_x
    else:  # val/test splits must not leak into mu: build a temporary train dataset
        Data = data_dict[args.data]
        X = Data(args=args, root_path=args.root_path, data_path=args.data_path, flag='train',
                 size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
                 target=args.target, timeenc=0 if args.embed != 'timeF' else 1,
                 freq=args.freq, seasonal_patterns=args.seasonal_patterns).data_x
    X = X[:12000]
    rng = np.random.default_rng(0)
    ch = rng.choice(X.shape[1], size=min(50, X.shape[1]), replace=False)
    return float(np.mean([_drift_index(X[:, j]) for j in ch]))


def _drift_to_lambda(drift, extrapolate=False):
    """Map drift index to equivalent Poisson shock rate via the synthetic calibration
    curve (logs/power_law/calibration.json); log-log linear interpolation, clamped.
    With extrapolate=True, a log-log line fit through the curve points is used and
    out-of-range drifts extrapolate instead of clamping to the nearest endpoint."""
    with open(POWER_LAW_CALIB) as f:
        curve = json.load(f)['_curve']
    pts = sorted((v['drift_mean'], v['lam']) for v in curve.values())
    log_d = np.log([p[0] for p in pts])
    log_l = np.log([p[1] for p in pts])
    if extrapolate:
        b, a = np.polyfit(log_d, log_l, 1)
        return float(np.exp(b * np.log(max(drift, 1e-6)) + a))
    return float(np.exp(np.interp(np.log(max(drift, 1e-6)), log_d, log_l)))


def _resolve_decay_auto_mu(args, data_set, flag):
    """drift index (train split) -> lambda (calibration) -> N* = C*lam^(-1/3)
    -> mu = exp(-1/N*). Cached on args so train/val/test share one value.

    Floor: mu >= exp(-decay_floor/L) i.e. N* >= L/decay_floor (default decay_floor=3,
    <=0 disables) — the forgotten effective window keeps at least 1/3 of the nominal
    window. Without it, datasets whose drift exceeds the calibration range clamp to
    N*~242, and on instance-normalized inverted-embedding models (iTransformer) that
    shrinkage poisons training (exchange 0.49 vs none 0.32). --decay_c overrides the
    synthetic power-law constant; --decay_extrapolate replaces endpoint clamping of
    the drift->lambda map with log-log extrapolation (real-data recalibration path).
    """
    mu = getattr(args, '_decay_mu_resolved', None)
    if mu is not None:
        return mu
    drift = _train_split_drift(args, data_set, flag)
    lam = _drift_to_lambda(drift, extrapolate=getattr(args, 'decay_extrapolate', False))
    c = getattr(args, 'decay_c', -1.0)
    c = POWER_LAW_C if c <= 0 else c
    nstar = c * lam ** (-1.0 / 3.0)
    mu = float(np.exp(-1.0 / nstar))
    floor_div = getattr(args, 'decay_floor', 3.0)
    if floor_div > 0:
        mu_floor = float(np.exp(-floor_div / args.seq_len))
        if mu < mu_floor:
            print(f'decay_auto: mu={mu:.6f} (N*={nstar:.0f}) below floor mu={mu_floor:.6f} '
                  f'(N*=L/{floor_div:g}={int(args.seq_len / floor_div)}), clamping')
            mu = mu_floor
    args._decay_mu_resolved = mu
    print(f'decay_auto: drift={drift:.4f} -> lambda={lam:.6f} -> N*={-1.0 / np.log(mu):.0f} -> mu={mu:.6f} '
          f'(c={c:g}, extrapolate={getattr(args, "decay_extrapolate", False)}, '
          f'floor={floor_div:g})')
    return mu


class InputDecayWrapper(Dataset):
    """Exponential forgetting of the encoder input (EHS drift-matched decay).

    Multiplies seq_x along time by w(t) = mu**(L-1-t): weight 1 at the most recent
    step, mu**(L-1) at the oldest. Soft alternative to the zero-fill hard budget
    (InputBudgetWrapper): far history stays nonzero, so instance-normalized inverted
    embedding (iTransformer) does not see a zero-dominated window (zero-fill poisons
    iTransformer: exchange 0.600 vs none 0.321, logs/spectral_budget). seq_y and
    time marks pass through unchanged; mu=1.0 is the identity; shape unchanged.
    Applied to every split. Activated by --input_decay exp (+ --decay_mu mu, or
    --decay_auto to resolve mu from the train-split drift index).
    """

    def __init__(self, base, seq_len, mu):
        self.base = base
        self.mu = mu
        self._w_cache = {}

    def __getattr__(self, name):  # delegate scaler/inverse_transform/etc.
        return getattr(self.base, name)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.base[index]
        L = seq_x.shape[0]
        w = self._w_cache.get(L)
        if w is None:
            w = self.mu ** np.arange(L - 1, -1, -1, dtype=np.float64)
            self._w_cache[L] = w
        seq_x = seq_x * w[:, None]  # `*` allocates; underlying data_x view untouched
        return seq_x, seq_y, seq_x_mark, seq_y_mark


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        ib = getattr(args, 'input_budget', -1)
        sb = getattr(args, 'spectral_budget', 'none')
        idec = getattr(args, 'input_decay', 'none')
        if idec == 'exp':  # exponential forgetting; mutually exclusive with budget wrappers
            mu = getattr(args, 'decay_mu', 0.999)
            if getattr(args, 'decay_auto', False):
                mu = _resolve_decay_auto_mu(args, data_set, flag)
            if mu < 1.0:
                data_set = InputDecayWrapper(data_set, args.seq_len, mu)
                print(f'input decay: exp forgetting mu={mu:.6f} '
                      f'(equivalent window N*={-1.0 / np.log(mu):.0f})')
            else:
                print('input decay: mu=1.0 (identity)')
        elif sb is not None and sb != 'none' and ib is not None and 0 < ib < args.seq_len:
            data_set = InputSpectralBudgetWrapper(data_set, args.seq_len, ib, sb, args.data_path)
            print(f'spectral budget [{sb}]: input steps older than the most recent '
                  f'{ib}/{args.seq_len} transformed, recent segment kept')
        elif ib is not None and 0 < ib < args.seq_len:
            data_set = InputBudgetWrapper(data_set, args.seq_len, ib)
            print(f'input budget: keeping most recent {ib}/{args.seq_len} steps, rest zeroed')
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
