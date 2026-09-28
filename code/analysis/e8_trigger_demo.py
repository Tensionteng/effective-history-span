#!/usr/bin/env python
"""E8: accumulated-evidence (CUSUM) changepoint trigger demo on synthetic
regime-switch series (review ccfa-review-reports/ehs-review.md E8; decides the
fig4/\xc2\xa76 trigger branch).

Data: tsfm/shock_data.py ShockTS (length 512, 0-3 shocks of 4 types with
ground-truth positions). Val seed 777 (n=1024) tunes the CUSUM thresholds by F1;
test seed 999 (n=2048, same protocol as Appendix G probes) is the held-out set.

Trigger: two-sided CUSUM on signed standardized one-step innovations
(allowance k=0.5) + one-sided CUSUM on squared innovations (variance path,
allowance kv=1.0); fire on either path crossing its threshold; 16-step
refractory. Innovations come from the constant-decay online AR(K) predictor
(pass A), i.e. the deployed model's own error stream; thresholds frozen after
val tuning.

Prediction demo (EWRLS, exponential forgetting via discounted sufficient
statistics of an AR(K=48) map; ridge 1.0; 1-step and direct 24-step):
  A const mu_law   : mu = exp(-1/N*), N* = c*lam^(-1/3), c=65.63 (paper's law,
                     lam = 1.5/512 for this generator) -- the paper's decay arm
  B const mu=0.99  : stronger constant forgetting (N*=100)
  C const mu=1.0   : no forgetting
  D trigger-reset  : mu_law + hard reset (S,b <- 0) one step after each trigger
  E oracle-reset   : mu_law + hard reset one step after each TRUE shock
Metrics: trigger precision/recall/F1 vs ground truth (hit window [-16,+48]),
mean delay, per-type recall, false alarms per shock-free series; MSE overall /
post-shock (target within 48 after a shock) / calm (shock-free series) /
stable (shocked series, >=96 after every shock), on the common target grid
[88, 512).

Two-pass disclosure: pass A computes innovations under constant mu_law; arm D
re-runs with resets at the triggers found in pass A (not a closed loop on arm
D's own innovations).

Outputs: logs/ehs_fix/e8/e8_trigger_demo.{json,md}
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tsfm.shock_data import ShockTS, SHOCK_NAMES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTD = os.path.join(ROOT, 'logs', 'ehs_fix', 'e8')
os.makedirs(OUTD, exist_ok=True)

L = 512
K = 48          # AR lag order
H = 24          # direct multi-step horizon (target = origin + H)
RIDGE = 1.0
BURN = 64       # MSE burn-in
GRID0 = BURN + H  # common target-grid start (88)
POST = 48       # post-shock region half-length
REFRACT = 16
C_LAW = 65.63   # power-law constant, logs/power_law (slope fixed -1/3)
SHOCK_RATE = 1.5 / 512.0  # mean shocks per step of the generator
NSTAR = C_LAW * SHOCK_RATE ** (-1.0 / 3.0)
MU_LAW = float(np.exp(-1.0 / NSTAR))

EARLY, LATE = 16, 48  # hit window for trigger<->shock matching


def load_series(seed, n):
    ds = ShockTS(n_series=n, length=L, seed=seed)
    X = np.zeros((n, L))
    shocks = []
    for i in range(n):
        b = ds[i]
        X[i] = b['x'].numpy()
        m = int(b['n'])
        shocks.append([(int(b['pos'][j]), int(b['typ'][j])) for j in range(m)])
    return X, shocks


def run_pass(X, mu, reset_steps=None, need_innov=False, reset_keep=0.0):
    """EWRLS walk-forward with intercept-augmented AR(K) features.
    reset_steps: list per series of steps at whose START S,b are scaled by
    reset_keep (0.0 = hard reset, 0.1 = soft reset keeping 10%).
    Returns 1-step innovations/errors and 24-step errors (indexed by target)."""
    N, T = X.shape
    KK = K + 1  # + intercept
    S1 = np.zeros((N, KK, KK))
    b1 = np.zeros((N, KK))
    S24 = np.zeros((N, KK, KK))
    b24 = np.zeros((N, KK))
    e1 = np.full((N, T), np.nan)
    sq1 = np.full((N, T), np.nan)
    sq24 = np.full((N, T), np.nan)
    IK = np.eye(KK) * RIDGE
    one = np.ones((N, 1))
    rmap = {}
    if reset_steps is not None:
        for i, steps in enumerate(reset_steps):
            for s in steps:
                rmap.setdefault(s, []).append(i)
    for t in range(K, T):
        if t in rmap:
            idx = rmap[t]
            S1[idx] *= reset_keep
            b1[idx] *= reset_keep
            S24[idx] *= reset_keep
            b24[idx] *= reset_keep
        phi_tm1 = np.concatenate([X[:, t - 1::-1][:, :K], one], axis=1)  # ends at t-1
        th1 = np.linalg.solve(S1 + IK, b1[..., None])[..., 0]
        y1 = (phi_tm1 * th1).sum(1)
        e = X[:, t] - y1
        if need_innov:
            e1[:, t] = e
        sq1[:, t] = e ** 2
        phi_t = np.concatenate([X[:, t::-1][:, :K], one], axis=1)        # ends at t
        th24 = np.linalg.solve(S24 + IK, b24[..., None])[..., 0]
        y24 = (phi_t * th24).sum(1)
        if t + H < T:
            sq24[:, t + H] = (X[:, t + H] - y24) ** 2   # indexed by target
        S1 *= mu
        S1 += phi_tm1[:, :, None] * phi_tm1[:, None, :]
        b1 *= mu
        b1 += phi_tm1 * X[:, t:t + 1]
        S24 *= mu
        b24 *= mu
        if t - H >= K - 1:
            phi_lag = np.concatenate([X[:, t - H::-1][:, :K], one], axis=1)
            S24 += phi_lag[:, :, None] * phi_lag[:, None, :]
            b24 += phi_lag * X[:, t:t + 1]
    return dict(e1=e1, sq1=sq1, sq24=sq24)


def cusum_triggers(e1, h, hv, k=0.5, kv=1.0, refract=REFRACT, t0=K + 8):
    """Vectorized two-path CUSUM over causal EMA-standardized innovations."""
    N, T = e1.shape
    Sp = np.zeros(N)
    Sn = np.zeros(N)
    Sv = np.zeros(N)
    last = np.full(N, -10 ** 9)
    rms = np.ones(N)
    trig = [[] for _ in range(N)]
    for t in range(K, T):
        e = e1[:, t]
        rms = 0.98 * rms + 0.02 * e * e
        if t < t0:
            continue
        eh = e / np.maximum(np.sqrt(rms), 0.2)
        Sp = np.maximum(0.0, Sp + eh - k)
        Sn = np.minimum(0.0, Sn + eh + k)
        Sv = np.maximum(0.0, Sv + eh * eh - 1.0 - kv)
        fire = ((Sp > h) | (Sn < -h) | (Sv > hv)) & (t - last >= refract)
        if fire.any():
            idx = np.nonzero(fire)[0]
            for i in idx:
                trig[i].append(t)
            Sp[idx] = 0.0
            Sn[idx] = 0.0
            Sv[idx] = 0.0
            last[idx] = t
    return trig


def cusum3_triggers(e1, hf=1e9, hs=1e9, hv=1e9, kf=0.5, ks=0.15, kv=1.0,
                    refract=REFRACT, t0=K + 8):
    """Three-path CUSUM: fast signed (kf=0.5), slow signed (ks small, for
    trend-level persistent bias), variance (kv). Fires on any path."""
    N, T = e1.shape
    Spf = np.zeros(N)
    Snf = np.zeros(N)
    Sps = np.zeros(N)
    Sns = np.zeros(N)
    Sv = np.zeros(N)
    last = np.full(N, -10 ** 9)
    rms = np.ones(N)
    trig = [[] for _ in range(N)]
    for t in range(K, T):
        e = e1[:, t]
        rms = 0.98 * rms + 0.02 * e * e
        if t < t0:
            continue
        eh = e / np.maximum(np.sqrt(rms), 0.2)
        Spf = np.maximum(0.0, Spf + eh - kf)
        Snf = np.minimum(0.0, Snf + eh + kf)
        Sps = np.maximum(0.0, Sps + eh - ks)
        Sns = np.minimum(0.0, Sns + eh + ks)
        Sv = np.maximum(0.0, Sv + eh * eh - 1.0 - kv)
        fire = ((Spf > hf) | (Snf < -hf) | (Sps > hs) | (Sns < -hs) | (Sv > hv)) \
            & (t - last >= refract)
        if fire.any():
            idx = np.nonzero(fire)[0]
            for i in idx:
                trig[i].append(t)
            Spf[idx] = 0.0
            Snf[idx] = 0.0
            Sps[idx] = 0.0
            Sns[idx] = 0.0
            Sv[idx] = 0.0
            last[idx] = t
    return trig


def pointwise_triggers(e1, tau, refract=REFRACT, t0=K + 8):
    """Pointwise-surprise baseline: fire on |standardized innovation| > tau."""
    N, T = e1.shape
    last = np.full(N, -10 ** 9)
    rms = np.ones(N)
    trig = [[] for _ in range(N)]
    for t in range(K, T):
        e = e1[:, t]
        rms = 0.98 * rms + 0.02 * e * e
        if t < t0:
            continue
        eh = e / np.maximum(np.sqrt(rms), 0.2)
        fire = (np.abs(eh) > tau) & (t - last >= refract)
        if fire.any():
            idx = np.nonzero(fire)[0]
            for i in idx:
                trig[i].append(t)
            last[idx] = t
    return trig


def match_metrics(trig, shocks):
    """Hit window [-EARLY, +LATE]; greedy nearest, each trigger/shock used once."""
    tp = fp = fn = 0
    delays = []
    per_type = {ty: [0, 0] for ty in range(4)}  # ty -> [hits, total]
    n_trig = n_shock = 0
    for tr, sh in zip(trig, shocks):
        used_t = set()
        n_trig += len(tr)
        n_shock += len(sh)
        for si, (s, ty) in enumerate(sh):
            per_type[ty][1] += 1
            cands = [(abs(t - s), ti, t) for ti, t in enumerate(tr)
                     if ti not in used_t and s - EARLY <= t <= s + LATE]
            if cands:
                cands.sort()
                _, ti, t = cands[0]
                used_t.add(ti)
                tp += 1
                delays.append(t - s)
                per_type[ty][0] += 1
            else:
                fn += 1
        fp += len(tr) - len(used_t)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn,
                triggers=n_trig, shocks=n_shock,
                delay_mean=float(np.mean(delays)) if delays else None,
                delay_median=float(np.median(delays)) if delays else None,
                per_type_recall={SHOCK_NAMES[ty]: per_type[ty][0] / max(1, per_type[ty][1])
                                 for ty in range(4)},
                per_type_n={SHOCK_NAMES[ty]: per_type[ty][1] for ty in range(4)})


def region_masks(shocks, T):
    N = len(shocks)
    grid = np.arange(GRID0, T)
    post = np.zeros((N, T), bool)
    stable = np.zeros((N, T), bool)
    calm = np.zeros((N, T), bool)
    for i, sh in enumerate(shocks):
        if not sh:
            calm[i, GRID0:] = True
            continue
        for s, _ in sh:
            lo, hi = max(s, GRID0), min(s + POST, T)
            post[i, lo:hi] = True
        stable[i, GRID0:] = True
        for s, _ in sh:
            lo, hi = max(s - 0, GRID0), min(s + 96, T)
            stable[i, lo:hi] = False
    overall = np.zeros((N, T), bool)
    overall[:, GRID0:] = True
    return dict(overall=overall, post=post, calm=calm, stable=stable)


def mse_region(sq, mask):
    """Per-series nanmean over masked positions, then mean over series with any."""
    vals = np.where(mask, sq, np.nan)
    cnt = np.sum(~np.isnan(vals), axis=1)
    tot = np.nansum(vals, axis=1)
    ok = cnt > 0
    return float(np.mean(tot[ok] / cnt[ok])), int(ok.sum())


def main():
    t0 = time.time()
    v2 = len(sys.argv) > 1 and sys.argv[1] == 'v2'
    tag = '_v2' if v2 else ''
    print(f'MU_LAW={MU_LAW:.6f} (N*={NSTAR:.0f}) mode={"v2" if v2 else "v1"}', flush=True)
    Xv, sh_v = load_series(777, 1024)
    Xt, sh_t = load_series(999, 2048)

    # ---- pass A on val + test (constant mu_law, collect innovations) ----
    pa_v = run_pass(Xv, MU_LAW, need_innov=True)
    print(f'pass A val done ({time.time()-t0:.0f}s)', flush=True)
    pa_t = run_pass(Xt, MU_LAW, need_innov=True)
    print(f'pass A test done ({time.time()-t0:.0f}s)', flush=True)

    if not v2:
        # ---- v1: two-path CUSUM tuning on val ----
        grid_h = [3.0, 4.0, 5.0, 6.0, 8.0, 1e9]
        grid_hv = [4.0, 6.0, 8.0, 12.0, 16.0, 1e9]
        tune = []
        for h in grid_h:
            for hv in grid_hv:
                tr = cusum_triggers(pa_v['e1'], h, hv)
                m = match_metrics(tr, sh_v)
                tune.append(dict(h=h, hv=hv, f1=m['f1'], precision=m['precision'],
                                 recall=m['recall'],
                                 trig_per_series=m['triggers'] / len(sh_v)))
                print(f'  tune h={h:g} hv={hv:g}: F1={m["f1"]:.3f} P={m["precision"]:.3f} '
                      f'R={m["recall"]:.3f} trig/seq={m["triggers"]/len(sh_v):.2f}', flush=True)
        tune.sort(key=lambda r: (-r['f1'], r['h'] + r['hv'] * 1e-3))
        best = tune[0]
        h_star, hv_star = best['h'], best['hv']
        print(f'chosen: h={h_star:g} hv={hv_star:g} (val F1={best["f1"]:.3f})', flush=True)
        tr_t = cusum_triggers(pa_t['e1'], h_star, hv_star)
        det = match_metrics(tr_t, sh_t)
        det['h'] = h_star
        det['hv'] = hv_star
        extra_det = {}
    else:
        # ---- v2: three-path CUSUM (fast/slow/variance) + pointwise baseline ----
        tune = []
        for hf in [4.0, 5.0, 6.0, 8.0, 1e9]:
            for ks in [0.15, 0.25]:
                for hs in [3.0, 4.0, 5.0, 6.0, 8.0, 1e9]:
                    for hv in [6.0, 10.0, 14.0, 1e9]:
                        tr = cusum3_triggers(pa_v['e1'], hf=hf, hs=hs, hv=hv, ks=ks)
                        m = match_metrics(tr, sh_v)
                        tune.append(dict(hf=hf, ks=ks, hs=hs, hv=hv, f1=m['f1'],
                                         precision=m['precision'], recall=m['recall'],
                                         trig_per_series=m['triggers'] / len(sh_v),
                                         trend_recall=m['per_type_recall']['trend'],
                                         freq_recall=m['per_type_recall']['freq']))
                        print(f'  tune hf={hf:g} ks={ks:g} hs={hs:g} hv={hv:g}: F1={m["f1"]:.3f} '
                              f'P={m["precision"]:.3f} R={m["recall"]:.3f} '
                              f'trend={m["per_type_recall"]["trend"]:.2f} '
                              f'freq={m["per_type_recall"]["freq"]:.2f}', flush=True)
        tune.sort(key=lambda r: (-r['f1'], r['hf'] + r['hs'] + r['hv'] * 1e-3))
        best = tune[0]
        print(f'chosen: {best}', flush=True)
        tr_t = cusum3_triggers(pa_t['e1'], hf=best['hf'], ks=best['ks'], hs=best['hs'], hv=best['hv'])
        det = match_metrics(tr_t, sh_t)
        det.update(best)
        # fast-only ablation (best cell with slow+variance paths off)
        fast_only = max((r for r in tune if r['hs'] > 1e8 and r['hv'] > 1e8),
                        key=lambda r: r['f1'])
        tr_fo = cusum3_triggers(pa_t['e1'], hf=fast_only['hf'])
        det_fast = match_metrics(tr_fo, sh_t)
        det_fast.update(fast_only)
        # pointwise baseline: tau tuned on val
        pw = []
        for tau in [2.0, 2.5, 3.0, 3.5, 4.0]:
            tr = pointwise_triggers(pa_v['e1'], tau)
            m = match_metrics(tr, sh_v)
            pw.append(dict(tau=tau, f1=m['f1']))
        pw.sort(key=lambda r: -r['f1'])
        tr_pw = pointwise_triggers(pa_t['e1'], pw[0]['tau'])
        det_pw = match_metrics(tr_pw, sh_t)
        det_pw['tau'] = pw[0]['tau']
        extra_det = dict(fast_only=det_fast, pointwise=det_pw)
        print(f'fast-only ablation: {det_fast}', flush=True)
        print(f'pointwise baseline: {det_pw}', flush=True)

    fa_free = [len(tr_t[i]) for i, sh in enumerate(sh_t) if not sh]
    det['fa_per_shockfree_series'] = float(np.mean(fa_free))
    det['n_shockfree'] = len(fa_free)
    print(f'test detection: {det}', flush=True)

    # ---- prediction arms on test ----
    trig_resets = [[t + 1 for t in tr] for tr in tr_t]
    orac_resets = [[s + 1 for s, _ in sh] for sh in sh_t]
    arms = {'A_const_mu_law': run_pass(Xt, MU_LAW),
            'B_const_mu0.99': run_pass(Xt, 0.99),
            'C_const_mu1.0': run_pass(Xt, 1.0),
            'D_trigger_reset': run_pass(Xt, MU_LAW, reset_steps=trig_resets),
            'E_oracle_reset': run_pass(Xt, MU_LAW, reset_steps=orac_resets)}
    if v2:
        arms['F_oracle_soft0.1'] = run_pass(Xt, MU_LAW, reset_steps=orac_resets, reset_keep=0.1)
        arms['G_trigger_soft0.1'] = run_pass(Xt, MU_LAW, reset_steps=trig_resets, reset_keep=0.1)
    print(f'arms done ({time.time()-t0:.0f}s)', flush=True)
    arms['A_const_mu_law'] = dict(sq1=pa_t['sq1'], sq24=pa_t['sq24'], e1=None)

    masks = region_masks(sh_t, L)
    res = {}
    for name, pa in arms.items():
        res[name] = {}
        for hz in ['sq1', 'sq24']:
            res[name][hz] = {}
            for rg, mk in masks.items():
                m, nser = mse_region(pa[hz], mk)
                res[name][hz][rg] = dict(mse=m, n_series=nser)
    out = dict(mu_law=MU_LAW, nstar=NSTAR, K=K, H=H, ridge=RIDGE, burn=BURN,
               grid0=GRID0, hit_window=[-EARLY, LATE], refract=REFRACT,
               intercept=True, mode='v2' if v2 else 'v1',
               val=dict(seed=777, n=1024), test=dict(seed=999, n=2048),
               tune=tune, chosen=best, detection=det, extra_detectors=extra_det, arms=res)

    # ---- verdict numbers ----
    def g(arm, hz, rg):
        return res[arm][hz][rg]['mse']
    verdict = {}
    for hz, lab in [('sq1', '1-step'), ('sq24', '24-step direct')]:
        v = dict(
            trigger_vs_const_overall=(g('A_const_mu_law', hz, 'overall') - g('D_trigger_reset', hz, 'overall'))
            / g('A_const_mu_law', hz, 'overall'),
            trigger_vs_const_post=(g('A_const_mu_law', hz, 'post') - g('D_trigger_reset', hz, 'post'))
            / g('A_const_mu_law', hz, 'post'),
            trigger_vs_const_calm=(g('D_trigger_reset', hz, 'calm') - g('A_const_mu_law', hz, 'calm'))
            / g('A_const_mu_law', hz, 'calm'),
            oracle_vs_const_overall=(g('A_const_mu_law', hz, 'overall') - g('E_oracle_reset', hz, 'overall'))
            / g('A_const_mu_law', hz, 'overall'),
            oracle_vs_const_post=(g('A_const_mu_law', hz, 'post') - g('E_oracle_reset', hz, 'post'))
            / g('A_const_mu_law', hz, 'post'),
            best_const_overall=min(g('A_const_mu_law', hz, 'overall'), g('B_const_mu0.99', hz, 'overall'),
                                   g('C_const_mu1.0', hz, 'overall')),
            trigger_overall=g('D_trigger_reset', hz, 'overall'),
        )
        if v2:
            v['oracle_soft_vs_const_overall'] = ((g('A_const_mu_law', hz, 'overall') - g('F_oracle_soft0.1', hz, 'overall'))
                                                 / g('A_const_mu_law', hz, 'overall'))
            v['oracle_soft_vs_const_post'] = ((g('A_const_mu_law', hz, 'post') - g('F_oracle_soft0.1', hz, 'post'))
                                              / g('A_const_mu_law', hz, 'post'))
            v['trigger_soft_vs_const_overall'] = ((g('A_const_mu_law', hz, 'overall') - g('G_trigger_soft0.1', hz, 'overall'))
                                                  / g('A_const_mu_law', hz, 'overall'))
            v['trigger_soft_vs_const_post'] = ((g('A_const_mu_law', hz, 'post') - g('G_trigger_soft0.1', hz, 'post'))
                                               / g('A_const_mu_law', hz, 'post'))
        verdict[lab] = v
    out['verdict'] = verdict

    with open(os.path.join(OUTD, f'e8_trigger_demo{tag}.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)

    # ---- markdown ----
    if v2:
        det_head = (f"## Detection (test; 3-path CUSUM hf={best['hf']:g} ks={best['ks']:g} "
                    f"hs={best['hs']:g} hv={best['hv']:g} frozen from val F1={best['f1']:.3f})")
    else:
        det_head = f"## Detection (test, h={h_star:g}, hv={hv_star:g} frozen from val F1)"
    lines = ['# E8 trigger demo (synthetic regime-switch)', '',
             f'mu_law={MU_LAW:.6f} (N*={NSTAR:.0f} from the paper power law, c=65.63, '
             f'lam=1.5/512); AR(K={K})+intercept EWRLS, ridge={RIDGE}; val seed 777 (n=1024, '
             f'threshold tuning), test seed 999 (n=2048, held out).',
             f'Hit window [{-EARLY},+{LATE}], refractory {REFRACT}; target grid [{GRID0},{L}).', '',
             det_head, '',
             f"- precision={det['precision']:.3f}, recall={det['recall']:.3f}, F1={det['f1']:.3f} "
             f"(tp={det['tp']}, fp={det['fp']}, fn={det['fn']}; triggers={det['triggers']}, shocks={det['shocks']})",
             f"- delay: mean={det['delay_mean']:.1f} steps, median={det['delay_median']:.1f}",
             f"- per-type recall: " + ', '.join(f"{k}={v:.3f} (n={det['per_type_n'][k]})"
                                                for k, v in det['per_type_recall'].items()),
             f"- false alarms per shock-free series: {det['fa_per_shockfree_series']:.3f} "
             f"(n={det['n_shockfree']})", '']
    if v2 and extra_det:
        for nm, dd in [('fast-only CUSUM ablation', extra_det['fast_only']),
                       ('pointwise-surprise baseline', extra_det['pointwise'])]:
            lines += [f"- {nm}: P={dd['precision']:.3f} R={dd['recall']:.3f} F1={dd['f1']:.3f}; "
                      f"per-type recall: " + ', '.join(f"{k}={v:.3f}" for k, v in dd['per_type_recall'].items())]
        lines.append('')
    lines += ['## Prediction MSE (per-series mean then series mean)', '']
    for hz, lab in [('sq1', '1-step'), ('sq24', '24-step direct')]:
        lines += [f'### {lab}', '',
                  '| arm | overall | post-shock | calm | stable |',
                  '|---|---|---|---|---|']
        for name in arms:
            r = res[name][hz]
            lines.append(f"| {name} | {r['overall']['mse']:.4f} | {r['post']['mse']:.4f} | "
                         f"{r['calm']['mse']:.4f} | {r['stable']['mse']:.4f} |")
        v = verdict[lab]
        lines += ['',
                  f"- trigger vs const(mu_law): overall {v['trigger_vs_const_overall']:+.2%}, "
                  f"post-shock {v['trigger_vs_const_post']:+.2%}, calm cost {v['trigger_vs_const_calm']:+.2%}",
                  f"- oracle(hard) vs const(mu_law): overall {v['oracle_vs_const_overall']:+.2%}, "
                  f"post-shock {v['oracle_vs_const_post']:+.2%}",
                  f"- trigger overall {v['trigger_overall']:.4f} vs best constant-arm overall "
                  f"{v['best_const_overall']:.4f}"]
        if v2:
            lines += [f"- oracle(soft0.1) vs const: overall {v['oracle_soft_vs_const_overall']:+.2%}, "
                      f"post-shock {v['oracle_soft_vs_const_post']:+.2%}",
                      f"- trigger(soft0.1) vs const: overall {v['trigger_soft_vs_const_overall']:+.2%}, "
                      f"post-shock {v['trigger_soft_vs_const_post']:+.2%}"]
        lines.append('')
    with open(os.path.join(OUTD, f'e8_trigger_demo{tag}.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'all done ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
