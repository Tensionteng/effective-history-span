"""Evaluation for the latent-action probe.

Metrics per arm (val set: ShockTS seed=999, 2048 series):
  - MSE: all positions / calm (shock-free series) / post-shock (shock observed,
    predict continuation) / pre-shock (predicting into a shock, unpredictable ref)
  - alignment (fsq only): code-change events vs shock times, precision/recall
    within +-1 patch; NMI/purity of fired code id vs shock type (on hits)
  - intervention (fsq only): force codes at the shock patch to the calm-mode code,
    measure whether predictions revert toward the no-shock counterfactual
Also writes matplotlib figures (series+codes, intervention, code-type heatmap,
ETTh1 sanity) to out_dir.
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tsfm.shock_data import ShockTS, SHOCK_NAMES, LEVEL, VOL, TREND, FREQ
from tsfm.latent_action import LatentActionModel

VAL_SEED = 999
N_VAL = 2048
COLORS = ['tab:red', 'tab:orange', 'tab:green', 'tab:purple']


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck['args']
    model = LatentActionModel(arm=a['arm'], patch=a['patch'], d_model=a['d_model'],
                              layers=a['layers'], heads=a['heads'], dec_layers=a['dec_layers'],
                              horizon=a['horizon'], max_len=a['seq_len'],
                              fsq_levels=tuple(a['fsq_levels']), latent_dim=a['latent_dim'],
                              shift_res=a.get('shift_res', False),
                              action_input=a.get('action_input', 'state'),
                              null_target=a.get('null_target', 0.0),
                              loss_type=a.get('loss_type', 'mse'),
                              gate_mode=a.get('gate_mode', 'abs'),
                              gate_thresh=a.get('gate_thresh', 0.0),
                              pred_mlp=a.get('pred_mlp', False),
                              code_feat=a.get('code_feat', 'raw'),
                              gate_accum=a.get('gate_accum', False)).to(device)
    model.load_state_dict(ck['model'], strict=False)  # older ckpts may lack newer buffers
    model.eval()
    return model, a


@torch.no_grad()
def collect(model, device, n_val=N_VAL, seed=VAL_SEED, bs=128, noise_lo=0.2, noise_hi=0.8):
    ds = ShockTS(n_series=n_val, seed=seed, noise_lo=noise_lo, noise_hi=noise_hi)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)
    out = {k: [] for k in ('pred', 'codes', 'x', 'base', 'pos', 'typ', 'n', 'p')}
    for b in dl:
        x = b['x'].to(device)
        pred, codes, q, p, logvar, aux = model(x)
        out['pred'].append(pred.cpu().numpy())
        out['p'].append(p.cpu().numpy())
        if codes is not None:
            out['codes'].append(codes.cpu().numpy())
        for k in ('x', 'base', 'pos', 'typ', 'n'):
            out[k].append(b[k].numpy())
    has_codes = len(out['codes']) > 0
    for k in out:
        if len(out[k]) > 0:
            out[k] = np.concatenate(out[k])
    if not has_codes:
        out['codes'] = None
    return out


def per_position_err(out, horizon=6, patch=16):
    """Returns err [N, L-H]: MSE of the horizon*patch prediction at each position."""
    B, L, P = out['p'].shape
    tgt = np.stack([out['p'][:, i + 1:i + 1 + horizon].reshape(B, horizon * P)
                    for i in range(L - horizon)], axis=1)
    return ((out['pred'][:, :L - horizon] - tgt) ** 2).mean(-1), tgt


def shock_patches(out):
    """Per sample: list of (patch, type) for valid shocks."""
    res = []
    for b in range(len(out['n'])):
        ev = []
        for j in range(int(out['n'][b])):
            ev.append((int(out['pos'][b][j]) // 16, int(out['typ'][b][j]), int(out['pos'][b][j])))
        res.append(ev)
    return res


def mse_metrics(out, horizon=6):
    err, _ = per_position_err(out, horizon)
    L_H = err.shape[1]
    evs = shock_patches(out)
    m = {}
    m['mse_all'] = float(err.mean())
    calm_mask = np.array([len(e) == 0 for e in evs])
    m['mse_calm'] = float(err[calm_mask].mean())
    post, pre = [], []
    for b, ev in enumerate(evs):
        for (p, t, s) in ev:
            if p >= L_H:
                continue
            for i in range(p, min(p + 3, L_H)):       # shock observed -> post-shock forecast
                post.append(err[b, i])
            for i in range(max(0, p - horizon), p):   # window contains shock, not yet observed
                if i < L_H and i + 1 <= p <= i + horizon:
                    pre.append(err[b, i])
    m['mse_post_shock'] = float(np.mean(post))
    m['mse_pre_shock'] = float(np.mean(pre))
    return m


def nmi_purity(ids, labels):
    ids, labels = np.asarray(ids), np.asarray(labels)
    N = len(ids)
    if N == 0:
        return 0.0, 0.0, {}
    ct = {}
    for c, l in zip(ids, labels):
        ct.setdefault(c, {})
        ct[c][l] = ct[c].get(l, 0) + 1
    pc = {c: sum(v.values()) / N for c, v in ct.items()}
    pl = {}
    for v in ct.values():
        for l, n in v.items():
            pl[l] = pl.get(l, 0) + n / N
    I = 0.0
    for c, v in ct.items():
        for l, n in v.items():
            p = n / N
            I += p * np.log(p / (pc[c] * pl[l]))
    Hc = -sum(p * np.log(p) for p in pc.values())
    Hl = -sum(p * np.log(p) for p in pl.values())
    nmi = 2 * I / (Hc + Hl + 1e-12)
    purity = sum(max(v.values()) for v in ct.values()) / N
    return float(nmi), float(purity), ct


def alignment_metrics(out, codebook_size=125):
    codes = out['codes']  # [N, L]
    N, L = codes.shape
    events = codes[:, 1:] != codes[:, :-1]  # event at position i (1..L-1) -> idx i-1
    evs = shock_patches(out)
    hits, total, fired_codes, fired_types = 0, 0, [], []
    tp_events, n_events = 0, 0
    for b in range(N):
        ev_pos = np.where(events[b])[0] + 1
        shocks = evs[b]
        n_events += len(ev_pos)
        for e in ev_pos:
            if any(abs(e - p) <= 1 for (p, t, s) in shocks):
                tp_events += 1
        for (p, t, s) in shocks:
            total += 1
            w = [e for e in ev_pos if abs(e - p) <= 1]
            if w:
                hits += 1
                e = min(w, key=lambda e: abs(e - p))
                fired_codes.append(int(codes[b, e]))
                fired_types.append(t)
    m = {}
    m['recall'] = hits / max(1, total)
    m['precision'] = tp_events / max(1, n_events)
    m['events_per_seq'] = n_events / N
    m['null_rate'] = float((codes == 0).mean())
    m['codes_used'] = int(len(np.unique(codes)))
    m['codebook_size'] = int(codebook_size)
    calm_mask = np.array([len(e) == 0 for e in evs])
    vals, counts = np.unique(codes[calm_mask], return_counts=True)
    m['calm_code'] = int(vals[np.argmax(counts)])
    m['calm_code_share'] = float(counts.max() / counts.sum())
    nmi, pur, ct = nmi_purity(fired_codes, fired_types)
    m['nmi_hit'] = nmi
    m['purity_hit'] = pur
    # regime-code test: code two patches after the shock vs shock type
    post_codes, post_types = [], []
    for b in range(N):
        for (p, t, s) in evs[b]:
            if p + 2 < L:
                post_codes.append(int(codes[b, p + 2]))
                post_types.append(t)
    m['nmi_post'], m['purity_post'], _ = nmi_purity(post_codes, post_types)
    m['n_hits'] = hits
    m['n_shocks'] = total
    m['_fired_codes'], m['_fired_types'], m['_contingency'] = fired_codes, fired_types, ct
    # per-type recall
    for t in range(4):
        n_t = sum(1 for (_, tt, _) in sum(evs, []) if tt == t)
        h_t = sum(1 for (c, tt) in zip(fired_codes, fired_types) if tt == t)
        m[f'recall_{SHOCK_NAMES[t]}'] = h_t / max(1, n_t)
    return m


@torch.no_grad()
def intervention_metrics(model, out, device, horizon=6, patch=16):
    """Force codes near shock patches to the calm code; compare predictions with the
    no-shock counterfactual and the true shocked series."""
    codes = torch.from_numpy(out['codes'])
    # calm code = mode over positions of shock-free sequences only
    calm_mask = np.array([len(e) == 0 for e in shock_patches(out)])
    vals, counts = np.unique(out['codes'][calm_mask], return_counts=True)
    calm = int(vals[np.argmax(counts)])
    L = codes.shape[1]
    L_H = L - horizon
    xs, cints, poss, fired = [], [], [], []
    for b, ev in enumerate(shock_patches(out)):
        for (p, t, s) in ev:
            if p >= L_H:
                continue
            ci = codes[b].clone()
            ci[max(0, p - 1):p + 2] = calm
            xs.append(torch.from_numpy(out['x'][b]))
            cints.append(ci)
            poss.append((b, p, t))
            fired.append(bool((codes[b][max(0, p - 1):p + 2] != calm).any()))
    res = dict(orig_true=[], orig_base=[], interv_true=[], interv_base=[], shock_size=[], per_type={})
    bs = 256
    for k in range(0, len(xs), bs):
        xb = torch.stack(xs[k:k + bs]).to(device)
        cb = torch.stack(cints[k:k + bs]).to(device)
        pred_o, _, _, _, _, _ = model(xb)
        pred_i, _, _, _, _, _ = model(xb, intervene_codes=cb)
        for r, (b, p, t) in enumerate(poss[k:k + bs]):
            st, en = (p + 1) * patch, (p + 1 + horizon) * patch
            true = out['x'][b][st:en]
            base = out['base'][b][st:en]
            po = pred_o[r, p].cpu().numpy()
            pi = pred_i[r, p].cpu().numpy()
            res['orig_true'].append(((po - true) ** 2).mean())
            res['orig_base'].append(((po - base) ** 2).mean())
            res['interv_true'].append(((pi - true) ** 2).mean())
            res['interv_base'].append(((pi - base) ** 2).mean())
            res['shock_size'].append(((true - base) ** 2).mean())
            res['per_type'].setdefault(t, [[], []])
            res['per_type'][t][0].append(((po - true) ** 2).mean())
            res['per_type'][t][1].append(((pi - true) ** 2).mean())
    m = {k: float(np.mean(v)) for k, v in res.items() if k != 'per_type'}
    m['per_type'] = {SHOCK_NAMES[t]: {'orig_true': float(np.mean(a)), 'interv_true': float(np.mean(b))}
                     for t, (a, b) in res['per_type'].items()}
    # same aggregates restricted to shocks where the gate actually fired (non-null
    # code near the shock) — the meaningful intervention subset for sparse models
    fired = np.array(fired, dtype=bool)
    m['fired_share'] = float(fired.mean())
    if fired.any():
        for k in ('orig_true', 'interv_true', 'orig_base', 'interv_base', 'shock_size'):
            m[k + '_fired'] = float(np.mean(np.array(res[k])[fired]))
    m['calm_code'] = calm
    m['n_interventions'] = len(xs)
    return m


def plot_examples(out, out_dir, n_show=3):
    codes = out['codes']
    evs = shock_patches(out)
    picks = [b for b in range(len(evs)) if len(evs[b]) >= 2][:n_show]
    fig, axes = plt.subplots(len(picks), 2, figsize=(13, 2.4 * len(picks)),
                             gridspec_kw={'width_ratios': [3, 1]})
    for row, b in enumerate(picks):
        ax, ax2 = axes[row]
        x = out['x'][b]
        ax.plot(x, lw=0.8, color='0.3')
        for (p, t, s) in evs[b]:
            ax.axvline(s, color=COLORS[t], lw=1.2, ls='--',
                       label=SHOCK_NAMES[t] if row == 0 else None)
        ax.set_title(f'sample {b} (shocks: {", ".join(SHOCK_NAMES[t] for _, t, _ in evs[b])})',
                     fontsize=9)
        ax.legend(fontsize=7, loc='upper right') if row == 0 else None
        ax2.step(np.arange(len(codes[b])), codes[b], where='post', lw=1.0, color='tab:blue')
        for (p, t, s) in evs[b]:
            ax2.axvline(p, color=COLORS[t], lw=1.2, ls='--')
        ax2.set_title('FSQ code / patch', fontsize=9)
        ax2.set_xlim(0, len(codes[b]))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_examples.png'), dpi=130)
    plt.close(fig)


def plot_intervention(model, out, device, out_dir):
    codes = torch.from_numpy(out['codes'])
    calm_mask = np.array([len(e) == 0 for e in shock_patches(out)])
    vals, counts = np.unique(out['codes'][calm_mask], return_counts=True)
    calm = int(vals[np.argmax(counts)])
    patch, horizon = 16, 6
    # pick a level-shock sample with p <= 20 where the gate actually fired
    for b, ev in enumerate(shock_patches(out)):
        cand = [(p, t, s) for (p, t, s) in ev
                if t == LEVEL and p <= 20 and codes[b][p] != calm]
        if cand:
            p, t, s = cand[0]
            break
    else:
        for b, ev in enumerate(shock_patches(out)):
            cand = [(p, t, s) for (p, t, s) in ev if t == LEVEL and p <= 20]
            if cand:
                p, t, s = cand[0]
                break
    xb = torch.from_numpy(out['x'][b:b + 1]).to(device)
    ci = codes[b].clone()
    ci[max(0, p - 1):p + 2] = calm
    with torch.no_grad():
        pred_o, co, _, _, _, _ = model(xb)
        pred_i, _, _, _, _, _ = model(xb, intervene_codes=ci.unsqueeze(0).to(device))
    st, en = (p + 1) * patch, (p + 1 + horizon) * patch
    fig, ax = plt.subplots(figsize=(11, 3.2))
    tt = np.arange(len(out['x'][b]))
    ax.plot(tt, out['x'][b], lw=0.9, color='0.3', label='observed (shocked)')
    ax.plot(tt, out['base'][b], lw=0.9, color='tab:green', label='no-shock counterfactual')
    w = np.arange(st, en)
    ax.plot(w, pred_o[0, p].cpu().numpy(), lw=1.4, color='tab:red',
            label=f'pred @ patch {p} (orig code {int(co[0, p])})')
    ax.plot(w, pred_i[0, p].cpu().numpy(), lw=1.4, color='tab:blue',
            label=f'pred @ patch {p} (code forced to calm {calm})')
    ax.axvline(s, color='tab:red', ls='--', lw=1.2, label='level shock')
    ax.axvspan(st, en, color='0.9', alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title(f'Intervention test, sample {b}: swap shock code -> calm code', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_intervention.png'), dpi=130)
    plt.close(fig)


def plot_code_type(am, out_dir, top_k=12):
    fired, types = np.array(am['_fired_codes']), np.array(am['_fired_types'])
    vals, counts = np.unique(fired, return_counts=True)
    top = vals[np.argsort(-counts)[:top_k]]
    M = np.zeros((len(top), 4))
    for i, c in enumerate(top):
        for t in range(4):
            M[i, t] = ((fired == c) & (types == t)).sum()
    Mn = M / (M.sum(1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    im = ax.imshow(Mn, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(4), SHOCK_NAMES)
    ax.set_yticks(range(len(top)), [f'{c} (n={int(counts[vals.tolist().index(c)])})' for c in top],
                  fontsize=8)
    for i in range(len(top)):
        for t in range(4):
            if M[i, t] > 0:
                ax.text(t, i, int(M[i, t]), ha='center', va='center', fontsize=7,
                        color='white' if Mn[i, t] > 0.5 else 'black')
    ax.set_title(f'Fired code x shock type (NMI={am["nmi_hit"]:.3f}, purity={am["purity_hit"]:.3f})',
                 fontsize=10)
    fig.colorbar(im, label='row fraction')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_code_type.png'), dpi=130)
    plt.close(fig)


def plot_etth1(model, device, out_dir, n_win=2):
    import pandas as pd
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'ETT-small', 'ETTh1.csv')
    ot = pd.read_csv(csv_path)['OT'].values.astype(np.float32)
    # rank windows by rolling-mean jump (level-shift-like), center the jump in the window
    L, roll = 512, 48
    jump = np.abs(np.convolve(ot, np.r_[np.ones(roll) / roll, -np.ones(roll) / roll], mode='same'))
    jump[:L] = 0
    jump[-L:] = 0
    starts = []
    for _ in range(n_win):
        j = int(np.argmax(jump))
        s = max(0, min(len(ot) - L, j - L // 2))
        starts.append(s)
        jump[max(0, s - L):s + 2 * L] = 0
    fig, axes = plt.subplots(n_win, 2, figsize=(13, 2.4 * n_win),
                             gridspec_kw={'width_ratios': [3, 1]})
    for row, s in enumerate(starts):
        w = ot[s:s + L]
        xb = torch.from_numpy(w[None]).to(device)
        with torch.no_grad():
            _, codes, _, _, _, _ = model(xb)
        codes = codes[0].cpu().numpy()
        axes[row][0].plot((w - w.mean()) / (w.std() + 1e-8), lw=0.8, color='0.3')
        axes[row][0].set_title(f'ETTh1 OT [{s}:{s + L}] (standardized)', fontsize=9)
        axes[row][1].step(np.arange(len(codes)), codes, where='post', lw=1.0, color='tab:blue')
        axes[row][1].set_title('FSQ code / patch (zero-shot)', fontsize=9)
        axes[row][1].set_xlim(0, len(codes))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_etth1.png'), dpi=130)
    plt.close(fig)


def evaluate(ckpt_path, device=None, out_dir='logs/latent_action', name=None, make_plots=True):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, a = load_model(ckpt_path, device)
    name = name or os.path.basename(ckpt_path).replace('ckpt_', '').replace('.pt', '')
    out = collect(model, device, noise_lo=a.get('noise_lo', 0.2), noise_hi=a.get('noise_hi', 0.8))
    m = {'arm': a['arm'], 'name': name, **mse_metrics(out)}
    if a['arm'] == 'fsq':
        am = alignment_metrics(out, codebook_size=model.fsq.codebook_size)
        m.update({k: v for k, v in am.items() if not k.startswith('_')})
        m['intervention'] = intervention_metrics(model, out, device)
        if make_plots:
            plot_examples(out, out_dir)
            plot_intervention(model, out, device, out_dir)
            plot_code_type(am, out_dir)
            plot_etth1(model, device, out_dir)
    json_path = os.path.join(out_dir, f'metrics_{name}.json')
    with open(json_path, 'w') as f:
        json.dump(m, f, indent=1)
    print(f'[eval {name}] ' + ' '.join(f'{k}={v:.4f}' for k, v in m.items()
                                      if isinstance(v, float)) + f' -> {json_path}')
    return m


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--out_dir', default='logs/latent_action')
    ap.add_argument('--name', default=None)
    args = ap.parse_args()
    print(json.dumps(evaluate(args.ckpt, out_dir=args.out_dir, name=args.name), indent=1))
