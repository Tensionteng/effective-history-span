"""Latent-action probe model: causal TS transformer + per-patch action head.

Three arms sharing the same backbone (d_model=128, 4 causal pre-norm layers,
patch=16, 512 steps -> 32 patches, predict next 6 patches = 96 steps):
  baseline: no action module (decoder reads encoder state directly)
  cont:     continuous 3-dim tanh bottleneck (Linear -> tanh -> Linear)
  fsq:      Finite Scalar Quantization bottleneck (Linear -> tanh -> round, STE)

The action path injects up(q) back into the encoder state residually; a 2-layer
causal decoder then predicts the next `horizon` patches from every position.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tsfm.model import RMSNorm, ResBlock


class FSQ(nn.Module):
    """Finite Scalar Quantization: per-dim tanh -> round to L levels, straight-through."""

    def __init__(self, levels=(5, 5, 5)):
        super().__init__()
        self.levels = list(levels)
        self.dim = len(levels)
        self.codebook_size = int(np.prod(levels))
        strides = np.cumprod([1] + self.levels[:-1])
        self.register_buffer('strides', torch.from_numpy(strides).long())
        self.register_buffer('half_lvl', (torch.tensor(self.levels, dtype=torch.float32) - 1) / 2)

    def forward(self, z):
        # z: [B, L, dim] -> quantized [B, L, dim] (STE), codes [B, L] long
        b = torch.tanh(z)
        r = torch.round(b * self.half_lvl)
        q = b + (r / self.half_lvl - b).detach()
        codes = ((r + self.half_lvl).long() * self.strides).sum(-1)
        return q, codes

    def id_to_vec(self, codes):
        # codes: [...] long -> quantized grid vectors [..., dim]
        idx = []
        rem = codes
        for s, l in zip(self.strides.tolist(), self.levels):
            idx.append(rem // s % l)
        idx = torch.stack(idx, dim=-1).float()
        return (idx - self.half_lvl) / self.half_lvl


class LatentActionModel(nn.Module):
    """v2 additions:
      action_input: 'state' (v1: code from encoder state z_i), 'innov' (code from
        residual patch_i - predictor(z_{i-1}), Genie/LAPA-style innovation), or
        'diff' (code from patch difference p_i - p_{i-1}). innov/diff are detached
        from the encoder so the action head cannot be gamed; the auxiliary
        predictor is trained on detached z with its own MSE.
      null_target>0: reserve code 0 as the null (no-event) code; positions whose
        innovation norm falls below an online quantile threshold (EMA of the
        null_target quantile) are forced to code 0 -> sparse firing.
      loss_type 'nll': extra head predicts per-step log-variance, Gaussian NLL
        (vol shocks become predictable -> vol codes have positive payoff)."""

    def __init__(self, arm='fsq', patch=16, d_model=128, layers=4, heads=8,
                 dec_layers=2, horizon=6, max_len=512, fsq_levels=(5, 5, 5), latent_dim=3,
                 shift_res=False, action_input='state', null_target=0.0, loss_type='mse',
                 gate_mode='abs', gate_thresh=0.0, pred_mlp=False, code_feat='raw',
                 gate_accum=False):
        super().__init__()
        assert arm in ('baseline', 'cont', 'fsq')
        assert action_input in ('state', 'innov', 'diff')
        assert loss_type in ('mse', 'nll')
        assert gate_mode in ('abs', 'white')
        assert code_feat in ('raw', 'rich')
        self.arm, self.patch, self.horizon = arm, patch, horizon
        # action bottleneck: at position i the decoder sees the raw state only up to
        # patch i-1; patch i enters solely through the (3-dim) action latent
        self.shift_res = shift_res and arm != 'baseline'
        self.action_input, self.null_target, self.loss_type = action_input, null_target, loss_type
        # gate_mode 'white': null gate fires on the scale-whitened innovation
        # (surprise relative to the predictor's own uncertainty), not raw magnitude
        self.gate_mode = gate_mode
        # gate_thresh > 0: absolute surprise threshold tau = gate_thresh * EMA-RMS
        # (firing rate emerges from data instead of being forced by a quantile)
        self.gate_thresh = gate_thresh
        # pred_mlp: 2-layer innovation predictor (better whitening -> less
        # sustained firing after vol shocks); code_feat 'rich': code input =
        # [raw residual, whitened residual, 4-step group norms]; gate_accum:
        # gate also fires on 0.8*||white_i + white_{i-1}|| (accumulated drift,
        # catches trend shocks whose per-patch surprise is weak)
        self.pred_mlp, self.code_feat, self.gate_accum = pred_mlp, code_feat, gate_accum
        self.use_null = null_target > 0 or gate_thresh > 0
        assert (not self.use_null) or (arm == 'fsq' and action_input != 'state')
        assert gate_mode == 'abs' or action_input == 'innov'
        assert code_feat == 'raw' or (action_input == 'innov' and gate_mode == 'white')
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len // patch, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([ResBlock(d_model, heads) for _ in range(layers)])
        self.final_norm = RMSNorm(d_model)
        feat_dim = patch * (2 if code_feat == 'rich' else 1) + (patch // 4 if code_feat == 'rich' else 0)
        in_dim = (d_model if action_input == 'state' else feat_dim)
        if action_input == 'innov':
            out_dim = 2 * patch if gate_mode == 'white' else patch
            if pred_mlp:
                self.predictor = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(),
                                               nn.Linear(128, out_dim))
            else:
                self.predictor = nn.Linear(d_model, out_dim)
        if arm == 'fsq':
            latent_dim = len(fsq_levels)  # FSQ dim must match the level list
            self.down = nn.Linear(in_dim, latent_dim)
            self.fsq = FSQ(fsq_levels)
            self.up = nn.Linear(latent_dim, d_model)
        elif arm == 'cont':
            self.down = nn.Linear(in_dim, latent_dim)
            self.up = nn.Linear(latent_dim, d_model)
        if self.use_null:
            self.register_buffer('tau', torch.tensor(-1.0))
            self.register_buffer('rms_ema', torch.tensor(-1.0))
        self.decoder = nn.ModuleList([ResBlock(d_model, heads) for _ in range(dec_layers)])
        self.head = nn.Linear(d_model, horizon * patch)
        if loss_type == 'nll':
            self.head_var = nn.Linear(d_model, horizon * patch)

    def encode(self, x):
        # x: [B, T] raw -> instance norm -> patches
        mu, sd = x.mean(1, keepdim=True), x.std(1, keepdim=True) + 1e-5
        xn = (x - mu) / sd
        L = xn.shape[1] // self.patch
        p = xn[:, :L * self.patch].unfold(1, self.patch, self.patch)  # [B, L, patch]
        z = self.embed(p) + self.pos[:, :L]
        for blk in self.blocks:
            z = blk(z)
        z = self.final_norm(z)
        return z, p

    def _shifted(self, t):
        t0 = torch.zeros_like(t[:, :1])
        return torch.cat([t0, t[:, :-1]], dim=1)

    def _resid(self, z):
        return self._shifted(z) if self.shift_res else z

    def action_features(self, z, p):
        """Returns (feat, gate_score, aux): feat feeds the action down-projection,
        gate_score [B, L] drives the null gate, aux trains the predictor.
        innov/diff features carry no encoder gradients by construction."""
        if self.action_input == 'state':
            return z, z.norm(dim=-1), None
        if self.action_input == 'innov':
            zp = self._shifted(z).detach()
            if self.gate_mode == 'white':
                mean, lv = self.predictor(zp).chunk(2, dim=-1)
                lv = lv.clamp(-8, 8)
                aux = 0.5 * (lv + (p - mean).pow(2) * torch.exp(-lv)).mean()
                raw = p - mean.detach()
                white = raw * torch.exp(-0.5 * lv.detach())
                score = white.norm(dim=-1)
                if self.gate_accum:
                    acc = (white + self._shifted(white)).norm(dim=-1) * 0.8
                    score = torch.maximum(score, acc)
                if self.code_feat == 'rich':
                    gn = raw.unflatten(-1, (self.patch // 4, 4)).norm(dim=-1)
                    feat = torch.cat([raw, white, gn], dim=-1)
                else:
                    feat = raw
                return feat, score, aux
            pred = self.predictor(zp)               # grads to predictor weights only
            aux = F.mse_loss(pred, p)
            raw = p - pred.detach()
            return raw, raw.norm(dim=-1), aux       # innovation, fully detached
        p_prev = self._shifted(p)
        diff = p - p_prev
        return diff, diff.norm(dim=-1), None

    def apply_null(self, q, codes, score):
        """Force low-surprise positions to the reserved null code 0.
        score: [B, L] precomputed gate score (higher = more surprising)."""
        norm = score.detach()
        if self.gate_thresh > 0:
            if self.training:
                r = norm.pow(2).mean().sqrt()
                self.rms_ema.copy_(r if self.rms_ema.item() < 0 else 0.99 * self.rms_ema + 0.01 * r)
            thresh = self.gate_thresh * self.rms_ema.clamp(min=1e-6)
        else:
            if self.training:
                t = torch.quantile(norm.flatten().float(), self.null_target)
                self.tau.copy_(t if self.tau.item() < 0 else 0.99 * self.tau + 0.01 * t)
            thresh = self.tau
        null = norm < thresh
        codes = torch.where(null, torch.zeros_like(codes), codes + 1)
        null_vec = self.fsq.id_to_vec(torch.zeros((), dtype=torch.long, device=q.device))
        q = torch.where(null.unsqueeze(-1), null_vec, q)
        return q, codes

    def code_to_vec(self, codes):
        # code 0 = null -> FSQ grid vector of id 0; code c > 0 -> FSQ id c-1
        return self.fsq.id_to_vec((codes - 1).clamp(min=0))

    def forward(self, x, intervene_codes=None):
        # intervene_codes: optional [B, L] long -> force these code ids (fsq arm)
        z, p = self.encode(x)
        aux = None
        if self.arm == 'baseline':
            h, q, codes = z, None, None
        else:
            feat, gate_score, aux = self.action_features(z, p)
            if intervene_codes is not None and self.arm == 'fsq':
                codes = intervene_codes.to(z.device)
                q = self.code_to_vec(codes) if self.use_null else self.fsq.id_to_vec(codes)
            else:
                a = self.down(feat)
                if self.arm == 'fsq':
                    q, codes = self.fsq(a)
                    if self.use_null:
                        q, codes = self.apply_null(q, codes, gate_score)
                else:
                    q, codes = torch.tanh(a), None
            h = self._resid(z) + self.up(q)
        for blk in self.decoder:
            h = blk(h)
        pred = self.head(h)  # [B, L, horizon*patch]
        logvar = self.head_var(h) if self.loss_type == 'nll' else None
        return pred, codes, q, p, logvar, aux

    def multi_patch_loss(self, pred, p, logvar=None, hmin=1):
        # p: [B, L, patch]; predict patches i+1..i+horizon from position i
        # hmin>1: loss only on horizon steps hmin..H (regime info matters more than
        # per-patch innovation at longer horizons -> action codes specialize)
        B, L, P = p.shape
        H = self.horizon
        tgt = torch.stack([p[:, i + 1:i + 1 + H].reshape(B, H * P)
                           for i in range(L - H)], dim=1)  # [B, L-H, H*P]
        pr = pred[:, :L - H].view(B, L - H, H, P)[:, :, hmin - 1:]
        tg = tgt.view(B, L - H, H, P)[:, :, hmin - 1:]
        if logvar is not None:
            lv = logvar[:, :L - H].view(B, L - H, H, P)[:, :, hmin - 1:].clamp(-8, 8)
            nll = 0.5 * (lv + (pr - tg).pow(2) * torch.exp(-lv)).mean()
            return nll, tgt
        return F.mse_loss(pr, tg), tgt
