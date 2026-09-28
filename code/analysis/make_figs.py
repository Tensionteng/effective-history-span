#!/usr/bin/env python
"""Publication figures + appendix tables for the EHS paper (paper/main.tex).

All numbers come from real experiment artifacts; nothing is hard-coded:

- fig1: controlled lookback sweep (iTransformer, equal training-data budget).
  Two panels: datasets where long context hurts (left) / helps (right).
  Y axis: test MSE relative to each dataset's L=96 arm. Diamond marks each
  dataset's best span. Source: logs/ehs_v2/SUMMARY.md (parsed).
- fig2: left  - drift index vs long-context benefit e(96)-e(2880), 16 samples
          (8 datasets x {iTransformer, DLinear}); Spearman recomputed.
        right - LOO ridge prediction of log2(best_sl) vs measured value,
          recomputed with the exact protocol of analysis/ehs_predict_v2.py
          (leave-one-dataset-out, top-3 features by |spearman| inside the
          training fold, ridge lam=1.0) and checked against
          logs/ehs_final/taskB_results.json.
        Sources: logs/ehs_final/taskB_samples.json, taskB_results.json.
- fig3: PatchTST layer-2 attention heatmaps on exchange_rate, L=96 vs
  L=2880, recomputed from the LBv2 checkpoints (eval mode, first 4 test
  batches, seeds 2021+2022 averaged, mean over channels/heads keeping the
  query axis, then column-normalized). Same forward path as
  analysis/mechanism/attn_entropy.py.

Also writes paper/appendix_tables.tex (DLinear/PatchTST/TimesNet full MSE
matrices from logs/ehs_v2/SUMMARY.md, saturation probe from
logs/ehs_final/taskA_sat.json, Chronos zero-shot table from SUMMARY.md).

Outputs: paper/figs/fig{1,2,3}.{pdf,png}, paper/appendix_tables.tex
Run: .venv/bin/python analysis/make_figs.py
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SUMMARY = os.path.join(ROOT, 'logs', 'ehs_v2', 'SUMMARY.md')
TASKB_SAMPLES = os.path.join(ROOT, 'logs', 'ehs_final', 'taskB_samples_train.json')
TASKB_RESULTS = os.path.join(ROOT, 'logs', 'ehs_final', 'taskB_results_train.json')
TASKA_SAT = os.path.join(ROOT, 'logs', 'ehs_final', 'taskA_sat.json')
ATTN_NPZ = os.path.join(ROOT, 'logs', 'mechanism', 'exp1', 'attn_metrics.npz')
FIGDIR = os.path.join(ROOT, 'paper', 'figs')
APPTEX = os.path.join(ROOT, 'paper', 'appendix_tables.tex')
os.makedirs(FIGDIR, exist_ok=True)

SEQ_LENS = [96, 336, 720, 1440, 2880]
DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather',
            'electricity', 'traffic']
DISPLAY = {'ETTh1': 'ETTh1', 'ETTh2': 'ETTh2', 'ETTm1': 'ETTm1', 'ETTm2': 'ETTm2',
           'exchange_rate': 'Exchange', 'weather': 'Weather',
           'electricity': 'Electricity', 'traffic': 'Traffic'}
# Okabe-Ito (yellow dropped for line visibility; black/gray used instead)
COLOR = {'exchange_rate': '#D55E00', 'ETTh1': '#56B4E9', 'ETTh2': '#0072B2',
         'ETTm1': '#009E73', 'ETTm2': '#E69F00', 'weather': '#CC79A7',
         'electricity': '#000000', 'traffic': '#999999'}
FIG1_LEFT = ['exchange_rate', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'ETTh1']
FIG1_RIGHT = ['electricity', 'traffic']

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6,
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})


def save(fig, name):
    for ext in ('pdf', 'png'):
        kw = dict(dpi=300) if ext == 'png' else {}
        fig.savefig(os.path.join(FIGDIR, f'{name}.{ext}'), **kw)
    plt.close(fig)
    print(f'[wrote] paper/figs/{name}.pdf/.png')


# ---------------------------------------------------------------- data parsing
CELL_RE = re.compile(r'^(?:\*\*)?([\d.]+)±([\d.]+)(?:\[n=(\d+)\])?(?:\*\*)?$')


def parse_mse_table(model):
    """Parse '### Model: <model> (MSE)' from SUMMARY.md.

    Returns vals[ds][sl] = (mean, std, n), best[ds] = sl (table's best column).
    """
    txt = open(SUMMARY).read()
    sec = re.search(rf'### Model: {model} \(MSE\)\n(.*?)(?:\n###|\n##|\Z)',
                    txt, re.S).group(1)
    vals, best = {}, {}
    for line in sec.splitlines():
        m = re.match(r'\| ([\w ]+?) \| (.*) \| (\d+) \|$', line.strip())
        if not m or m.group(1).strip() == 'dataset':
            continue
        ds, cells, b = m.group(1).strip(), m.group(2), int(m.group(3))
        vals[ds] = {}
        for sl, cell in zip(SEQ_LENS, cells.split(' | ')):
            cell = cell.strip()
            if cell == '-':
                continue
            cm = CELL_RE.match(cell)
            assert cm, f'unparsed cell: {cell!r}'
            mean, std, n = float(cm.group(1)), float(cm.group(2)), cm.group(3)
            vals[ds][sl] = (mean, std, int(n) if n else 3)
        best[ds] = b
    return vals, best


# ----------------------------------------------------------------------- fig1
def fig1():
    vals, best = parse_mse_table('iTransformer')
    for ds in DATASETS:  # the table's best column must equal the argmin
        argmin = min(vals[ds], key=lambda s: vals[ds][s][0])
        assert argmin == best[ds], (ds, argmin, best[ds])

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), sharex=True)
    for ax, group, title in ((axes[0], FIG1_LEFT, 'long context hurts'),
                             (axes[1], FIG1_RIGHT, 'long context helps')):
        for ds in group:
            sls = SEQ_LENS
            mean = np.array([vals[ds][s][0] for s in sls])
            std = np.array([vals[ds][s][1] for s in sls])
            rel, rel_std = mean / mean[0], std / mean[0]
            c = COLOR[ds]
            ax.plot(sls, rel, '-o', ms=2.8, color=c, label=DISPLAY[ds])
            ax.fill_between(sls, rel - rel_std, rel + rel_std, color=c,
                            alpha=0.15, lw=0)
            b = best[ds]
            ax.plot([b], [rel[sls.index(b)]], marker='D', ms=5.5,
                    mfc='white', mec=c, mew=1.4, zorder=5)
        ax.axhline(1.0, color='0.5', lw=0.7, ls='--', zorder=0)
        ax.set_xscale('log', base=2)
        ax.set_xticks(SEQ_LENS)
        ax.set_xticklabels([str(s) for s in SEQ_LENS])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(alpha=0.3, lw=0.4)
        ax.set_title(title)
        ax.set_xlabel('lookback $L$')
    axes[0].set_ylabel('relative test MSE (vs $L{=}96$)')
    # endpoint annotations (computed from the plotted values)
    v = vals['exchange_rate']
    axes[0].annotate(f"{v[2880][0] / v[96][0]:.1f}$\\times$",
                     xy=(2880, v[2880][0] / v[96][0]), xytext=(-26, 1),
                     textcoords='offset points', fontsize=6.5,
                     color=COLOR['exchange_rate'])
    for ds in FIG1_RIGHT:
        v = vals[ds]
        pct = (v[2880][0] / v[96][0] - 1) * 100
        axes[1].annotate(f'{pct:+.0f}%', xy=(2880, v[2880][0] / v[96][0]),
                         xytext=(-7, 4 if ds == 'electricity' else -5),
                         textcoords='offset points', fontsize=6.5, ha='right',
                         color=COLOR[ds])
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], marker='D', ls='', mfc='white', mec='0.2', ms=5.5))
    l.append('best span')
    axes[0].legend(h, l, loc='upper left', framealpha=0.9, edgecolor='0.8',
                   handlelength=1.4, borderpad=0.4, labelspacing=0.35)
    axes[1].legend(loc='upper right', framealpha=0.9, edgecolor='0.8',
                   handlelength=1.4, borderpad=0.4, labelspacing=0.35)
    axes[0].set_ylim(0.93, 3.95)
    axes[1].set_ylim(0.80, 1.03)
    fig.tight_layout(w_pad=1.6)
    save(fig, 'fig1_sweep')


# ----------------------------------------------------------------------- fig2
def loo_predict(samples):
    """Reproduce analysis/ehs_predict_v2.py LOO (k=3) exactly."""
    stats = ['periodicity_dt', 'drift', 'long_acf', 'calendar_acf_diff',
             'long_acf_diff']

    def ridge_fit(X, y, lam=1.0):
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Xs = (X - mu) / sd
        ymu = y.mean()
        A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
        w = np.linalg.solve(A, Xs.T @ (y - ymu))
        return w, mu, sd, ymu

    preds, trues = [], []
    for held in sorted({s['dataset'] for s in samples}):
        tr = [s for s in samples if s['dataset'] != held]
        te = [s for s in samples if s['dataset'] == held]
        ytr = np.array([s['log2_best_sl'] for s in tr])
        rank = sorted(stats,
                      key=lambda k: abs(spearmanr([s[k] for s in tr], ytr)[0]),
                      reverse=True)[:3]
        Xtr = np.array([[s[k] for k in rank] for s in tr])
        w, mu, sd, ymu = ridge_fit(Xtr, ytr)
        for s in te:
            x = np.array([s[k] for k in rank])
            preds.append(float((x - mu) / sd @ w + ymu))
            trues.append(float(s['log2_best_sl']))
    return np.array(preds), np.array(trues)


def fig2():
    samples = json.load(open(TASKB_SAMPLES))
    results = json.load(open(TASKB_RESULTS))
    assert len(samples) == 16

    drift = np.array([s['drift'] for s in samples])
    benefit = np.array([s['benefit'] for s in samples])
    rho_b, p_b = spearmanr(drift, benefit)

    preds, trues = loo_predict(samples)
    rho_l, p_l = spearmanr(preds, trues)
    stored = results['loo']['k3']['spearman'][0]
    assert abs(rho_l - stored) < 1e-3, (rho_l, stored)  # protocol check
    tier_hit = results['loo']['k3']['tier_hit']
    tiers = np.array(SEQ_LENS, dtype=float)
    med = float(np.median([s['log2_best_sl'] for s in samples]))
    const_tier = int(tiers[np.argmin(np.abs(tiers - 2 ** med))])
    base_hit = sum(int(const_tier == s['best_sl']) for s in samples)

    models = {'iTransformer': dict(marker='o'), 'DLinear': dict(marker='^')}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    ax = axes[0]
    for s in samples:
        ax.scatter(s['drift'], s['benefit'], s=16, color=COLOR[s['dataset']],
                   marker=models[s['model']]['marker'],
                   edgecolors='0.15', linewidths=0.5, zorder=3)
    ax.axhline(0.0, color='0.5', lw=0.7, ls='--', zorder=0)
    ax.set_xlabel('drift index (segment-mean variance ratio)')
    ax.set_ylabel('benefit $e(96)-e(2880)$ (MSE)')
    ax.set_title('drift predicts the sign of the benefit')
    ax.grid(alpha=0.3, lw=0.4)
    ax.text(0.03, 0.05, f'Spearman $\\rho={rho_b:.2f}$ ($p<10^{{-3}}$, $n=16$)',
            transform=ax.transAxes, fontsize=7)
    mk = [Line2D([], [], marker=models[m]['marker'], ls='', color='0.4',
                 mec='0.15', ms=4.5, label=m) for m in models]
    ax.legend(handles=mk, loc='upper right', framealpha=0.9, edgecolor='0.8',
              borderpad=0.4, handletextpad=0.2)

    ax = axes[1]
    dodge = {'iTransformer': -0.06, 'DLinear': 0.06}
    for s, p in zip(samples, preds):
        ax.scatter(s['log2_best_sl'] + dodge[s['model']], p, s=16,
                   color=COLOR[s['dataset']],
                   marker=models[s['model']]['marker'],
                   edgecolors='0.15', linewidths=0.5, zorder=3)
    lo, hi = np.log2(96) - 0.35, np.log2(2880) + 0.55
    ax.plot([lo, hi], [lo, hi], ls='--', color='0.5', lw=0.8, zorder=0)
    tiers = np.log2(SEQ_LENS)
    for t in tiers:
        ax.axvline(t, color='0.85', lw=0.4, zorder=0)
        ax.axhline(t, color='0.85', lw=0.4, zorder=0)
    ax.set_xticks(tiers)
    ax.set_xticklabels([str(s) for s in SEQ_LENS])
    ax.set_yticks(tiers)
    ax.set_yticklabels([str(s) for s in SEQ_LENS])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('measured best span $\\log_2 L^*$')
    ax.set_ylabel('LOO prediction $\\log_2 \\hat{L}^*$')
    ax.set_title('span is predictable before training')
    ax.text(0.97, 0.04,
            f'LOO Spearman $\\rho={rho_l:.3f}$ ($p={p_l:.3f}$)\n'
            f'exact tier {tier_hit}/16 (const. baseline {base_hit}/16)',
            transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    handles = [Line2D([], [], marker='s', ls='', color=COLOR[d], ms=4.5,
                      label=DISPLAY[d]) for d in DATASETS]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               handletextpad=0.15, columnspacing=0.9, borderpad=0.1)
    fig.tight_layout(w_pad=1.8, rect=(0, 0.09, 1, 1))
    save(fig, 'fig2_predictability')
    print(f'[fig2] drift-vs-benefit spearman={rho_b:+.4f} p={p_b:.2e}; '
          f'LOO spearman={rho_l:+.4f} p={p_l:.2e} (stored {stored:+.4f})')


# ----------------------------------------------------------------------- fig3
def fig3():
    import argparse
    import sys
    sys.path.insert(0, ROOT)
    import torch
    from data_provider.data_factory import data_provider
    from models import PatchTST as M

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    PATCH_LEN, STRIDE = 16, 8

    def build(sl):
        cfg = argparse.Namespace(
            task_name='long_term_forecast', seq_len=sl, pred_len=96,
            d_model=512, n_heads=8, e_layers=2, d_ff=2048, dropout=0.1,
            activation='gelu', enc_in=8, factor=3, attn_variant='none',
            recmask_window=336)
        model = M.Model(cfg).float().to(device).eval()
        for layer in model.encoder.attn_layers:
            layer.attention.inner_attention.output_attention = True
        return model

    def make_args(sl):
        return argparse.Namespace(
            task_name='long_term_forecast', is_training=0, model_id='attnprobe',
            model='PatchTST', data='custom', root_path='./dataset/exchange_rate/',
            data_path='exchange_rate.csv', features='M', target='OT', freq='h',
            checkpoints='./checkpoints/', seq_len=sl, label_len=48, pred_len=96,
            seasonal_patterns='Monthly', inverse=False, embed='timeF',
            batch_size=8, num_workers=0)

    @torch.no_grad()
    def attn_map(sl, seeds=(2021, 2022), n_batches=4):
        """Seed-averaged layer-2 attention map [Lq, S] + normalized entropy."""
        maps, ents = [], []
        for seed in seeds:
            hits = sorted(glob.glob(os.path.join(
                ROOT, 'checkpoints',
                f'long_term_forecast_LBv2_exchange_rate_PatchTST_{sl}_s{seed}_*',
                'checkpoint.pth')))
            assert hits, f'no checkpoint for sl={sl} seed={seed}'
            model = build(sl)
            sd = torch.load(hits[0], map_location=device)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            assert not unexpected, unexpected
            _, loader = data_provider(make_args(sl), 'test')
            acc, cnt = None, 0
            for i, (bx, by, bxm, bym) in enumerate(loader):
                if i >= n_batches:
                    break
                bx = bx.float().to(device)
                means = bx.mean(1, keepdim=True).detach()
                x = bx - means
                stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True,
                                             unbiased=False) + 1e-5)
                x = (x / stdev).permute(0, 2, 1)
                enc_out, _ = model.patch_embedding(x)
                _, attns = model.encoder(enc_out)
                A = attns[1]                      # second layer [B*C, H, Lq, S]
                A_ = A.clamp_min(1e-12)
                ents.append((-(A_ * A_.log()).sum(-1)).mean().item())
                acc = A.sum(0) if acc is None else acc + A.sum(0)
                cnt += A.shape[0]
                del attns, A
            maps.append((acc / cnt).mean(0).float().cpu().numpy())  # head-mean
            del model, acc
            torch.cuda.empty_cache()
        S = int((sl - PATCH_LEN) / STRIDE + 2)
        return np.mean(maps, 0), float(np.mean(ents)) / np.log(S)

    A96, e96 = attn_map(96)
    A2880, e2880 = attn_map(2880)
    # stored 8-batch probe values (the numbers quoted in the paper/caption)
    d = np.load(ATTN_NPZ, allow_pickle=True)
    st96 = float(np.mean([d[f'exchange_rate_96_s{s}_lbv2_entropy_norm'][1]
                          for s in (2021, 2022)]))
    st2880 = float(np.mean([d[f'exchange_rate_2880_s{s}_lbv2_entropy_norm'][1]
                            for s in (2021, 2022)]))
    print(f'[fig3] layer-2 normalized entropy: recomputed(4 batches) '
          f'{e96:.3f} -> {e2880:.3f}; stored(8 batches) {st96:.3f} -> {st2880:.3f}')

    def colnorm(A):
        return A / A.sum(axis=0, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6))
    for ax, A, sl in ((axes[0], A96, 96), (axes[1], A2880, 2880)):
        im = ax.imshow(colnorm(A), origin='lower', aspect='auto', cmap='viridis')
        ax.set_title(f'$L={sl}$ ($S={A.shape[0]}$ patches)')
        ax.set_xlabel('key patch (0 = oldest)')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label('column-normalized attention', fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
    axes[0].set_ylabel('query patch')
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig3_attention')


# ---------------------------------------------------------- appendix tables
def tex_cell(mean, std, n, is_best):
    m = f'{mean:.4f}'
    core = f'\\mathbf{{{m}}}' if is_best else m
    s = f'${core}\\pm{std:.4f}$'
    if n < 3:
        s += f'$^{{n{n}}}$'
    return s


def matrix_table(model, label):
    vals, best = parse_mse_table(model)
    rows = []
    for ds in DATASETS:
        cells = []
        for sl in SEQ_LENS:
            if sl in vals[ds]:
                m_, s_, n_ = vals[ds][sl]
                cells.append(tex_cell(m_, s_, n_, best[ds] == sl))
            else:
                cells.append('--')
        rows.append(f"{DISPLAY[ds]} & {' & '.join(cells)} & {best[ds]} \\\\")
    body = '\n'.join(rows)
    return rf"""\begin{{table}}[t]
\caption{{Full lookback sweep for {model} under the equal-data-budget protocol (test MSE, mean $\pm$ std over seeds 2021--2023; bold = best lookback of the row). All cells aggregate three seeds (Appendix ledger).}}
\label{{tab:{label}}}
\centering\footnotesize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lcccccc}}
\toprule
Dataset & $L{{=}}96$ & $L{{=}}336$ & $L{{=}}720$ & $L{{=}}1440$ & $L{{=}}2880$ & Best \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def sat_table():
    d = json.load(open(TASKA_SAT))['table']
    sls = [96, 336, 720, 1440, 2880, 5760]
    rows = []
    for ds in ['electricity', 'traffic']:
        for model in ['iTransformer', 'DLinear']:
            cells = []
            v = d[ds][model]
            best_sl = min(sls, key=lambda s: v[str(s)]['mean'])
            for sl in sls:
                c = v[str(sl)]
                cells.append(tex_cell(c['mean'], c['std'], c['n'],
                                      sl == best_sl))
            rows.append(f"{DISPLAY[ds]} & {model} & {' & '.join(cells)} \\\\")
        if ds == 'electricity':
            rows.append(r'\midrule')
    body = '\n'.join(rows)
    return rf"""\begin{{table}}[t]
\caption{{Saturation probe on electricity and traffic with $N_{{\min}}$ recomputed at $L{{=}}5760$ (test MSE, mean $\pm$ std). The $L{{=}}2880$ and $L{{=}}5760$ arms have three seeds each; shorter arms are single-seed ($^{{n1}}$). Errors at $L{{=}}5760$ never improve on $L{{=}}2880$, so the long-context benefit plateaus instead of being clipped by the scan range.}}
\label{{tab:sat}}
\centering\footnotesize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llcccccc}}
\toprule
Dataset & Model & $L{{=}}96$ & $L{{=}}336$ & $L{{=}}720$ & $L{{=}}1440$ & $L{{=}}2880$ & $L{{=}}5760$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def chronos_table():
    txt = open(SUMMARY).read()
    sec = re.search(r'## TSFM zero-shot.*?\n(.*?)(?:\n##|\Z)', txt, re.S).group(1)
    sls = [96, 336, 720, 1440, 2880]
    rows = []
    for line in sec.splitlines():
        m = re.match(r'\| ([\w ]+?) \| (.*) \|$', line.strip())
        if not m or m.group(1).strip() == 'dataset':
            continue
        ds, cells = m.group(1).strip(), m.group(2).split(' | ')
        out = []
        for cell in cells:
            cell = cell.strip()
            out.append('--' if cell == '-' else f'${cell}$')
        rows.append(f"{DISPLAY[ds]} & {' & '.join(out[:5])} \\\\")
    body = '\n'.join(rows)
    return rf"""\begin{{table}}[t]
\caption{{Zero-shot Chronos-Bolt (single run; test MSE\,/\,MAE). The data-side ordering persists without any training on our corpora: the ETT datasets and weather improve monotonically up to $L{{=}}1440$, while exchange rate is best at $L{{=}}96$. `--' marks arms cut by the time budget.}}
\label{{tab:chronos}}
\centering\footnotesize
\resizebox{{0.85\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccc}}
\toprule
Dataset & $L{{=}}96$ & $L{{=}}336$ & $L{{=}}720$ & $L{{=}}1440$ & $L{{=}}2880$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def write_appendix_tables():
    parts = [
        '% Auto-generated by analysis/make_figs.py from logs/ehs_v2/SUMMARY.md,',
        '% logs/ehs_final/taskA_sat.json. Do not edit numbers by hand.',
        '% Requires booktabs and graphicx (for \\resizebox).',
        matrix_table('DLinear', 'full_dlinear'),
        matrix_table('PatchTST', 'full_patchtst'),
        matrix_table('TimesNet', 'full_timesnet'),
        sat_table(),
        chronos_table(),
    ]
    with open(APPTEX, 'w') as f:
        f.write('\n'.join(parts))
    print(f'[wrote] paper/appendix_tables.tex')


def main():
    fig1()
    fig2()
    fig3()
    write_appendix_tables()
    print('[done]')


if __name__ == '__main__':
    main()
