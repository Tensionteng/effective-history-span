#!/usr/bin/env python
"""v2 hardening collector: W*(lambda) on the 10-tier ladder for two model families,
log-log fits for both estimators, and bootstrap CIs over seeds/tiers.

Reads logs/power_law/v2/{itransformer,dlinear}_drift_er*_sl*_s*.log, writes:
  logs/power_law/v2/verdict.json                    (per-family W* tables, fits, CIs, verdicts)
  logs/power_law/v2/figs/error_surface_{model}.png  (lambda x L MSE heatmap with W* marked)
  logs/power_law/v2/figs/mse_curves.png             (MSE vs lookback per lambda, both families)
  logs/power_law/v2/figs/powerlaw_fit.png           (log-log W* vs lambda, both families)
  logs/power_law/v2/figs/bootstrap_slopes.png       (slope bootstrap distributions)

Bootstrap: B replicates; each resamples the 10 tiers with replacement and, per sampled
tier, resamples the 2 seed-MSE-vectors with replacement, then recomputes W* (grid argmin
and parabola-interp separately) and the OLS slope. Both families consume the SAME draws
(paired), so the family slope difference gets a valid CI. Interp replicates where any
sampled tier is boundary-censored are skipped (count reported).

Run: .venv/bin/python analysis/power_law_collect_v2.py
"""
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOGDIR = os.path.join(ROOT, 'logs', 'power_law', 'v2')
FIG = os.path.join(LOGDIR, 'figs')
ER_PURE = [50, 150, 250, 400, 650, 1000, 1700, 2500, 4000, 6000]
LOOKBACKS = [96, 192, 336, 480, 720, 960, 1440, 2000, 2880]
SEEDS = [2021, 2022]
MODELS = ['itransformer', 'dlinear']
TARGET = -1.0 / 3.0       # v1-measured exponent family under test
TARGET_V1_THEORY = -2.0 / 3.0
TOL = 0.15
B = 10000
BOOT_SEED = 12345

LOG_RE = re.compile(r'^(itransformer|dlinear)_drift_er(\d+)_sl(\d+)_s(\d+)\.log$')
MSE_RE = re.compile(r'mse:([0-9.]+), mae:([0-9.]+)')


def collect():
    """res[model][(er, sl)][seed] = (mse, mae)"""
    res = {m: defaultdict(dict) for m in MODELS}
    for fn in sorted(os.listdir(LOGDIR)):
        mm = LOG_RE.match(fn)
        if not mm:
            continue
        model, er, sl, seed = mm.group(1), int(mm.group(2)), int(mm.group(3)), int(mm.group(4))
        path = os.path.join(LOGDIR, fn)
        with open(path, 'rb') as f:
            f.seek(max(0, os.path.getsize(f.name) - 4000))
            tail = f.read().decode(errors='ignore')
        found = None
        for m in MSE_RE.finditer(tail):
            found = (float(m.group(1)), float(m.group(2)))
        if found is not None:
            res[model][(er, sl)][seed] = found
    return res


def seed_mean_curve(res_m, er, seeds=SEEDS):
    curve = {}
    for sl in LOOKBACKS:
        vals = [res_m[(er, sl)][s][0] for s in seeds if s in res_m.get((er, sl), {})]
        if len(vals) == len(seeds):
            curve[sl] = float(np.mean(vals))
    return curve


def wstar_grid(curve):
    if len(curve) < len(LOOKBACKS):
        return None, True
    w = min(curve, key=curve.get)
    censored = w in (LOOKBACKS[0], LOOKBACKS[-1])
    return w, censored


def wstar_interp(curve):
    """Continuous W* via parabola in (ln L, MSE) through the 3 points bracketing
    the grid argmin; None at a grid boundary or non-convex fit."""
    pts = sorted(curve.items())
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


def fit_slope(er_w):
    xs = np.log(1.0 / np.array([e for e, _ in er_w], dtype=float))
    ys = np.log(np.array([w for _, w in er_w], dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    yhat = slope * xs + intercept
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(slope), float(intercept), float(r2)


def bootstrap(res_m, tier_draws, seed_draws):
    """tier_draws: (B, 10) tier indices; seed_draws: (B, 10, 2) seed indices.
    Returns per-replicate (grid slope, interp slope or nan), n_interp_skipped."""
    curves = {er: {s: {sl: res_m[(er, sl)][s][0] for sl in LOOKBACKS
                       if s in res_m.get((er, sl), {})} for s in SEEDS} for er in ER_PURE}
    slopes_g, slopes_i, n_skip = [], [], 0
    for b in range(len(tier_draws)):
        er_w_g, er_w_i, ok = [], [], True
        for j, ti in enumerate(tier_draws[b]):
            er = ER_PURE[ti]
            ss = [SEEDS[k] for k in seed_draws[b, j]]
            mean_curve = {sl: float(np.mean([curves[er][s][sl] for s in ss]))
                          for sl in LOOKBACKS if sl in curves[er][ss[0]] and sl in curves[er][ss[1]]}
            w, _ = wstar_grid(mean_curve)
            if w is None:
                ok = False
                break
            er_w_g.append((er, w))
            wi = wstar_interp(mean_curve)
            if wi is None:
                er_w_i.append(None)
            else:
                er_w_i.append((er, wi))
        if not ok:
            continue
        slopes_g.append(fit_slope(er_w_g)[0])
        if any(v is None for v in er_w_i):
            n_skip += 1
        else:
            slopes_i.append(fit_slope(er_w_i)[0])
    return np.array(slopes_g), np.array(slopes_i), n_skip


def ci(arr, lo=2.5, hi=97.5):
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def main():
    res = collect()
    n = {m: sum(len(v) for v in res[m].values()) for m in MODELS}
    print(f'collected arms: {n} (expect {len(ER_PURE) * len(LOOKBACKS) * len(SEEDS)} each)')

    tables = {}
    for m in MODELS:
        curves, wtab, cens, wint = {}, {}, {}, {}
        for er in ER_PURE:
            curves[er] = seed_mean_curve(res[m], er)
            w, c = wstar_grid(curves[er])
            wtab[er], cens[er] = w, c
            wint[er] = wstar_interp(curves[er])
        tables[m] = dict(curves=curves, wstar=wtab, censored=cens, wstar_interp=wint)

        print(f'\n=== {m} ===')
        print('MSE by lookback (2-seed mean):')
        print('| dataset | ' + ' | '.join(str(sl) for sl in LOOKBACKS) + ' | W* | W*_cont |')
        for er in ER_PURE:
            row = ' | '.join(f'{curves[er].get(sl, float("nan")):.4f}' for sl in LOOKBACKS)
            wi = wint[er]
            print(f'| drift_er{er} | {row} | {wtab[er]} | {wi and round(wi)} |')

        slope, intercept, r2 = fit_slope([(er, wtab[er]) for er in ER_PURE])
        iw = [(er, w) for er, w in wint.items() if w]
        slope_i, intercept_i, r2_i = fit_slope(iw)
        print(f'grid-argmin fit: slope={slope:.3f} intercept={intercept:.3f} R2={r2:.3f} '
              f'(target {TARGET:.3f}±{TOL}) -> {"PASS" if abs(slope - TARGET) <= TOL else "FAIL"}')
        print(f'interp-W* fit:   slope={slope_i:.3f} intercept={intercept_i:.3f} R2={r2_i:.3f} '
              f'-> {"PASS" if abs(slope_i - TARGET) <= TOL else "FAIL"}; '
              f'W*_cont=' + str({er: round(wint[er]) if wint[er] else None for er in ER_PURE}))
        for er in ER_PURE:
            per_seed = {}
            for s in SEEDS:
                c1 = {sl: res[m][(er, sl)][s][0] for sl in LOOKBACKS if s in res[m].get((er, sl), {})}
                per_seed[s] = min(c1, key=c1.get) if c1 else None
            print(f'  er={er:5d} lam={1 / er:.5f} W*={wtab[er]} censored={cens[er]} '
                  f'seed2021->{per_seed[SEEDS[0]]} seed2022->{per_seed[SEEDS[1]]}')
        tables[m].update(fit_grid=dict(slope=slope, intercept=intercept, r2=r2),
                         fit_interp=dict(slope=slope_i, intercept=intercept_i, r2=r2_i))

    # paired bootstrap over tiers x seeds
    rng = np.random.default_rng(BOOT_SEED)
    tier_draws = rng.integers(0, len(ER_PURE), size=(B, len(ER_PURE)))
    seed_draws = rng.integers(0, len(SEEDS), size=(B, len(ER_PURE), len(SEEDS)))
    boot = {}
    for m in MODELS:
        sg, si, n_skip = bootstrap(res[m], tier_draws, seed_draws)
        boot[m] = dict(
            grid=dict(median=float(np.median(sg)), ci=list(ci(sg)), n=len(sg)),
            interp=dict(median=float(np.median(si)), ci=list(ci(si)), n=len(si),
                        skipped=n_skip),
            slopes_grid=sg, slopes_interp=si)
        print(f'\n{m} bootstrap (B={B}, tiers x seeds resampled):')
        print(f'  grid:   median={np.median(sg):.3f} 95% CI [{ci(sg)[0]:.3f}, {ci(sg)[1]:.3f}]')
        print(f'  interp: median={np.median(si):.3f} 95% CI [{ci(si)[0]:.3f}, {ci(si)[1]:.3f}] '
              f'(skipped {n_skip} censored replicates)')
        for tgt, name in ((TARGET, '-1/3'), (TARGET_V1_THEORY, '-2/3')):
            in_g = ci(sg)[0] <= tgt <= ci(sg)[1]
            in_i = ci(si)[0] <= tgt <= ci(si)[1]
            print(f'  {name} inside grid CI: {in_g}; inside interp CI: {in_i}')
    # paired family difference (same draws)
    dg = boot['itransformer']['slopes_grid'] - boot['dlinear']['slopes_grid']
    print(f'\nfamily slope difference (iTransformer - DLinear, grid): '
          f'median={np.median(dg):.3f} 95% CI [{ci(dg)[0]:.3f}, {ci(dg)[1]:.3f}]')

    os.makedirs(FIG, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for m in MODELS:
        Z = np.full((len(ER_PURE), len(LOOKBACKS)), np.nan)
        for i, er in enumerate(ER_PURE):
            for j, sl in enumerate(LOOKBACKS):
                Z[i, j] = tables[m]['curves'][er].get(sl, np.nan)
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        im = ax.imshow(Z, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xticks(range(len(LOOKBACKS)), LOOKBACKS)
        ax.set_yticks(range(len(ER_PURE)), [f'1/{er}' for er in ER_PURE])
        ax.set_xlabel('lookback W')
        ax.set_ylabel('lambda')
        ax.set_title(f'v2 test MSE (pure drift, {m})')
        for i, er in enumerate(ER_PURE):
            j = LOOKBACKS.index(tables[m]['wstar'][er])
            ax.plot(j, i, 'r*', ms=14)
        fig.colorbar(im, label='MSE')
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f'error_surface_{m}.png'), dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, m in zip(axes, MODELS):
        for er in ER_PURE:
            curve = tables[m]['curves'][er]
            ax.plot(list(curve.keys()), list(curve.values()), 'o-', ms=3, label=f'1/{er}')
        ax.set_xscale('log')
        ax.set_xticks(LOOKBACKS, LOOKBACKS)
        ax.set_xlabel('lookback W')
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('test MSE')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'mse_curves.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = dict(itransformer='b', dlinear='g')
    for m in MODELS:
        wtab = tables[m]['wstar']
        xs = np.log(1.0 / np.array(ER_PURE, dtype=float))
        ys = np.log(np.array([wtab[er] for er in ER_PURE], dtype=float))
        fg = tables[m]['fit_grid']
        fi = tables[m]['fit_interp']
        ax.scatter(np.exp(xs), np.exp(ys), c=colors[m], zorder=3,
                   label=f'{m} W* grid (slope={fg["slope"]:.3f}, R2={fg["r2"]:.3f})')
        xl = np.linspace(xs.min() - 0.2, xs.max() + 0.2, 50)
        ax.plot(np.exp(xl), np.exp(fg['slope'] * xl + fg['intercept']), colors[m] + '-',
                alpha=0.7)
        wi = [tables[m]['wstar_interp'][er] for er in ER_PURE]
        if all(wi):
            ax.scatter(np.exp(xs), wi, facecolors='none', edgecolors=colors[m], s=70,
                       zorder=3, label=f'{m} W* interp (slope={fi["slope"]:.3f}, R2={fi["r2"]:.3f})')
        for i, er in enumerate(ER_PURE):
            flag = ' (censored)' if tables[m]['censored'][er] else ''
            ax.annotate(f'er{er}{flag}', (np.exp(xs[i]), np.exp(ys[i])), fontsize=7,
                        xytext=(4, 4), textcoords='offset points')
    xl = np.linspace(np.log(1 / 6000.0) - 0.2, np.log(1 / 50.0) + 0.2, 50)
    mid_intercept = tables['itransformer']['fit_grid']['intercept']
    ax.plot(np.exp(xl), np.exp(TARGET * xl + mid_intercept), 'k:', alpha=0.6,
            label=f'slope={TARGET:.3f} reference')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('lambda')
    ax.set_ylabel('W*')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'powerlaw_fit.png'), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axes, MODELS):
        ax.hist(boot[m]['slopes_grid'], bins=60, alpha=0.6, color=colors[m],
                label=f'grid median={np.median(boot[m]["slopes_grid"]):.3f}')
        ax.hist(boot[m]['slopes_interp'], bins=60, alpha=0.4, color='orange',
                label=f'interp median={np.median(boot[m]["slopes_interp"]):.3f}')
        ax.axvline(TARGET, c='k', ls=':', label='-1/3')
        ax.axvline(TARGET_V1_THEORY, c='r', ls=':', label='-2/3')
        ax.set_title(f'{m} bootstrap slopes (B={B})')
        ax.set_xlabel('log-log slope')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'bootstrap_slopes.png'), dpi=150)
    plt.close(fig)

    out = dict(
        config=dict(er_ladder=ER_PURE, lookbacks=LOOKBACKS, seeds=SEEDS, models=MODELS,
                    target=TARGET, tol=TOL, B=B, boot_seed=BOOT_SEED),
        families={m: dict(
            wstar={str(er): tables[m]['wstar'][er] for er in ER_PURE},
            wstar_interp={str(er): tables[m]['wstar_interp'][er] for er in ER_PURE},
            censored={str(er): tables[m]['censored'][er] for er in ER_PURE},
            curves={str(er): tables[m]['curves'][er] for er in ER_PURE},
            fit_grid={k: v for k, v in tables[m]['fit_grid'].items()},
            fit_interp={k: v for k, v in tables[m]['fit_interp'].items()},
            verdict_grid='PASS' if abs(tables[m]['fit_grid']['slope'] - TARGET) <= TOL else 'FAIL',
            verdict_interp='PASS' if abs(tables[m]['fit_interp']['slope'] - TARGET) <= TOL else 'FAIL',
            boot_grid=boot[m]['grid'], boot_interp=boot[m]['interp'],
            target_in_ci=dict(grid={'-1/3': bool(ci(boot[m]['slopes_grid'])[0] <= TARGET <= ci(boot[m]['slopes_grid'])[1]),
                                    '-2/3': bool(ci(boot[m]['slopes_grid'])[0] <= TARGET_V1_THEORY <= ci(boot[m]['slopes_grid'])[1])},
                              interp={'-1/3': bool(ci(boot[m]['slopes_interp'])[0] <= TARGET <= ci(boot[m]['slopes_interp'])[1]),
                                      '-2/3': bool(ci(boot[m]['slopes_interp'])[0] <= TARGET_V1_THEORY <= ci(boot[m]['slopes_interp'])[1])}),
        ) for m in MODELS},
        family_diff_grid=dict(median=float(np.median(dg)), ci=list(ci(dg))),
    )
    with open(os.path.join(LOGDIR, 'verdict.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('\nverdict.json + figs written to logs/power_law/v2/')


if __name__ == '__main__':
    main()
