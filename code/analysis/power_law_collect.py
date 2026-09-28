#!/usr/bin/env python
"""Collect power-law arms, compute W*(lambda), fit log-log slope, emit figs + verdict.

Reads logs/power_law/{drift,per}_er*_sl*_s*.log, writes:
  logs/power_law/verdict.json           (W* table, fit slope/intercept/R2, c stability)
  logs/power_law/figs/error_surface.png (lambda x L MSE heatmap with W* marked)
  logs/power_law/figs/mse_curves.png    (MSE vs lookback per lambda, pure + periodic)
  logs/power_law/figs/powerlaw_fit.png  (log-log W* vs lambda + OLS line)

Run: .venv/bin/python analysis/power_law_collect.py
"""
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOGDIR = os.path.join(ROOT, 'logs', 'power_law')
FIG = os.path.join(LOGDIR, 'figs')
ER_PURE = [50, 150, 400, 1000, 2500, 6000]
ER_PERIODIC = [150, 2500]
LOOKBACKS = [96, 192, 336, 720, 1440, 2880]
TARGET_SLOPE = -2.0 / 3.0
TOL = 0.15

LOG_RE = re.compile(r'^(drift|per)_er(\d+)_sl(\d+)_s(\d+)\.log$')
MSE_RE = re.compile(r'mse:([0-9.]+), mae:([0-9.]+)')


def collect():
    res = defaultdict(dict)
    for fn in sorted(os.listdir(LOGDIR)):
        m = LOG_RE.match(fn)
        if not m:
            continue
        tag, er, sl, seed = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        path = os.path.join(LOGDIR, fn)
        with open(path, 'rb') as f:
            f.seek(max(0, os.path.getsize(path) - 4000))
            tail = f.read().decode(errors='ignore')
        found = None
        for mm in MSE_RE.finditer(tail):
            found = (float(mm.group(1)), float(mm.group(2)))
        if found is not None:
            res[(tag, er, sl)][seed] = found
    return res


def wstar(res, tag, er):
    """argmin over lookback of seed-mean MSE; returns (W*, per-lookback means, censored)."""
    means = {}
    for sl in LOOKBACKS:
        vals = [v[0] for v in res.get((tag, er, sl), {}).values()]
        if vals:
            means[sl] = float(np.mean(vals))
    if not means:
        return None, {}, False
    w = min(means, key=means.get)
    censored = w in (LOOKBACKS[0], LOOKBACKS[-1])
    return w, means, censored


def wstar_interp(means):
    """Continuous W* via parabola in (ln L, MSE) through the 3 points bracketing
    the grid argmin. Returns None when the minimum is at a grid boundary."""
    pts = sorted((sl, m) for sl, m in means.items())
    xs = np.log([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    i = int(np.argmin(ys))
    if i == 0 or i == len(pts) - 1:
        return None
    x0, x1, x2 = xs[i - 1], xs[i], xs[i + 1]
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    if a <= 0:
        return None
    return float(np.exp(-b / (2 * a)))


def main():
    res = collect()
    n = sum(len(v) for v in res.values())
    print(f'collected {n} finished arms')

    wtab, curvetab, cens = {}, {}, {}
    for tag, ers in (('drift', ER_PURE), ('per', ER_PERIODIC)):
        for er in ers:
            w, means, c = wstar(res, tag, er)
            wtab[(tag, er)] = w
            curvetab[(tag, er)] = means
            cens[(tag, er)] = c

    print('\nMSE by lookback (2-seed mean):')
    header = '| dataset | ' + ' | '.join(str(sl) for sl in LOOKBACKS) + ' | W* |'
    print(header)
    for er in ER_PURE:
        means = curvetab[('drift', er)]
        row = ' | '.join(f'{means.get(sl, float("nan")):.4f}' for sl in LOOKBACKS)
        print(f'| drift_er{er} | {row} | {wtab[("drift", er)]} |')
    for er in ER_PERIODIC:
        means = curvetab[('per', er)]
        row = ' | '.join(f'{means.get(sl, float("nan")):.4f}' for sl in LOOKBACKS)
        print(f'| per_er{er} | {row} | {wtab[("per", er)]} |')

    # log-log fit on pure-drift W*
    def fit(wvals):
        xs = np.log(1.0 / np.array([er for er, w in wvals], dtype=float))
        ys = np.log(np.array([w for er, w in wvals], dtype=float))
        slope, intercept = np.polyfit(xs, ys, 1)
        yhat = slope * xs + intercept
        ss_res = float(np.sum((ys - yhat) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        cvals = {er: float(w * (1.0 / er) ** (2.0 / 3.0)) for er, w in wvals}
        c_arr = np.array(list(cvals.values()))
        return slope, intercept, r2, cvals, float(c_arr.std() / c_arr.mean())

    grid_wvals = [(er, wtab[('drift', er)]) for er in ER_PURE if wtab[('drift', er)]]
    slope, intercept, r2, cvals, c_cv = fit(grid_wvals)
    verdict = abs(slope - TARGET_SLOPE) <= TOL
    print(f'\ngrid-argmin fit: slope={slope:.3f} intercept={intercept:.3f} R2={r2:.3f} '
          f'(target {TARGET_SLOPE:.3f} ± {TOL}) -> {"PASS" if verdict else "FAIL"}')
    print('c = W* * lambda^(2/3) per level:', {k: round(v, 3) for k, v in cvals.items()},
          f'cv={c_cv:.3f}')

    # continuous-W* sensitivity: parabola interpolation in (ln L, MSE)
    interp_w = {}
    for er in ER_PURE:
        interp_w[er] = wstar_interp(curvetab[('drift', er)])
    interp_wvals = [(er, w) for er, w in interp_w.items() if w]
    slope_i, intercept_i, r2_i, cvals_i, c_cv_i = fit(interp_wvals)
    verdict_i = abs(slope_i - TARGET_SLOPE) <= TOL
    print(f'interpolated-W* fit: slope={slope_i:.3f} intercept={intercept_i:.3f} R2={r2_i:.3f} '
          f'-> {"PASS" if verdict_i else "FAIL"}; W*_cont='
          + str({er: round(w, 0) for er, w in interp_wvals}))
    print('c (interp) per level:', {k: round(v, 2) for k, v in cvals_i.items()}, f'cv={c_cv_i:.3f}')

    # per-seed argmin agreement
    print('\nper-seed W* (grid):')
    for er in ER_PURE:
        per_seed = {}
        for seed in (2021, 2022):
            means_s = {sl: res[('drift', er, sl)][seed][0] for sl in LOOKBACKS
                       if seed in res.get(('drift', er, sl), {})}
            per_seed[seed] = min(means_s, key=means_s.get) if means_s else None
        print(f'  er={er}: s2021->{per_seed.get(2021)} s2022->{per_seed.get(2022)} '
              f'agree={per_seed.get(2021) == per_seed.get(2022)}')

    for er in ER_PURE:
        w = wtab[('drift', er)]
        print(f'  er={er:5d} lam={1 / er:.5f} W*={w} censored={cens[("drift", er)]}')
    for er in ER_PERIODIC:
        print(f'  periodic er={er}: W*={wtab[("per", er)]} vs pure {wtab[("drift", er)]} '
              f'(shift {"UP" if (wtab[("per", er)] or 0) > (wtab[("drift", er)] or 0) else "no/ DOWN"})')

    os.makedirs(FIG, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # error surface (pure drift)
    Z = np.full((len(ER_PURE), len(LOOKBACKS)), np.nan)
    for i, er in enumerate(ER_PURE):
        for j, sl in enumerate(LOOKBACKS):
            Z[i, j] = curvetab[('drift', er)].get(sl, np.nan)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(Z, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xticks(range(len(LOOKBACKS)), LOOKBACKS)
    ax.set_yticks(range(len(ER_PURE)), [f'1/{er}' for er in ER_PURE])
    ax.set_xlabel('lookback W')
    ax.set_ylabel('lambda')
    ax.set_title('test MSE (pure drift)')
    for i, er in enumerate(ER_PURE):
        j = LOOKBACKS.index(wtab[('drift', er)])
        ax.plot(j, i, 'r*', ms=14)
    fig.colorbar(im, label='MSE')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'error_surface.png'), dpi=150)

    # MSE curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for er in ER_PURE:
        means = curvetab[('drift', er)]
        ax.plot(list(means.keys()), list(means.values()), 'o-', label=f'pure 1/{er}')
    for er in ER_PERIODIC:
        means = curvetab[('per', er)]
        ax.plot(list(means.keys()), list(means.values()), 's--', label=f'periodic 1/{er}')
    ax.set_xscale('log')
    ax.set_xticks(LOOKBACKS, LOOKBACKS)
    ax.set_xlabel('lookback W')
    ax.set_ylabel('test MSE')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'mse_curves.png'), dpi=150)

    # power-law fit
    xs = np.log(1.0 / np.array([er for er, w in grid_wvals], dtype=float))
    ys = np.log(np.array([w for er, w in grid_wvals], dtype=float))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(np.exp(xs), np.exp(ys), c='b', zorder=3, label='pure drift W* (grid argmin)')
    xl = np.linspace(xs.min() - 0.2, xs.max() + 0.2, 50)
    ax.plot(np.exp(xl), np.exp(slope * xl + intercept), 'b-',
            label=f'grid fit: slope={slope:.3f}, R2={r2:.3f}')
    if interp_wvals:
        xi = np.log(1.0 / np.array([er for er, w in interp_wvals], dtype=float))
        yi = np.log(np.array([w for er, w in interp_wvals], dtype=float))
        ax.scatter(np.exp(xi), np.exp(yi), facecolors='none', edgecolors='g', s=80,
                   zorder=3, label=f'W* interp (slope={slope_i:.3f}, R2={r2_i:.3f})')
    ax.plot(np.exp(xl), np.exp(TARGET_SLOPE * xl + intercept), 'k:', alpha=0.6,
            label=f'theory slope={TARGET_SLOPE:.3f}')
    for er in ER_PERIODIC:
        w = wtab[('per', er)]
        if w:
            ax.scatter([1.0 / er], [w], c='r', marker='s', zorder=3,
                       label=f'periodic 1/{er} W*')
    for i, (er, w) in enumerate(grid_wvals):
        flag = ' (censored)' if cens[('drift', er)] else ''
        ax.annotate(f'er{er}{flag}', (np.exp(xs[i]), np.exp(ys[i])), fontsize=7,
                    xytext=(4, 4), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('lambda')
    ax.set_ylabel('W*')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'powerlaw_fit.png'), dpi=150)

    out = dict(
        wstar={f'{tag}_er{er}': wtab[(tag, er)] for tag, er in
               [('drift', e) for e in ER_PURE] + [('per', e) for e in ER_PERIODIC]},
        wstar_interp={f'drift_er{er}': interp_w[er] for er in ER_PURE},
        censored={f'drift_er{er}': cens[('drift', er)] for er in ER_PURE},
        curves={f'{tag}_er{er}': curvetab[(tag, er)] for tag, er in
                [('drift', e) for e in ER_PURE] + [('per', e) for e in ER_PERIODIC]},
        fit_grid=dict(slope=float(slope), intercept=float(intercept), r2=float(r2),
                      target=TARGET_SLOPE, tol=TOL, verdict='PASS' if verdict else 'FAIL'),
        fit_interp=dict(slope=float(slope_i), intercept=float(intercept_i), r2=float(r2_i),
                        verdict='PASS' if verdict_i else 'FAIL'),
        c_per_level=cvals, c_cv=c_cv,
        c_per_level_interp=cvals_i, c_cv_interp=c_cv_i,
    )
    with open(os.path.join(LOGDIR, 'verdict.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('\nverdict.json + figs written')


if __name__ == '__main__':
    main()
