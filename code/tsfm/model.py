"""Minimal decoder-only TSFM backbone with selectable residual connection type.

Modes: res (standard pre-norm residual), hc (unconstrained hyper-connections),
mhc (Birkhoff-constrained, Sinkhorn), ohc (orthogonal, Cayley).
Next-patch prediction pretraining on raw scalar series.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.iTransformerHC import sinkhorn, cayley


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class HCConn(nn.Module):
    """Hyper-connection wrapper: x = alpha^T H (RMSNorm) -> y = F(x) -> H <- A H + beta * y."""

    def __init__(self, d, n, mode):
        super().__init__()
        self.n, self.mode = n, mode
        self.norm = RMSNorm(d)
        self.alpha = nn.Parameter(torch.zeros(n))
        self.alpha.data[0] = 1.0
        self.beta = nn.Parameter(torch.ones(n))
        if mode == 'mhc':
            self.theta = nn.Parameter(20.0 * torch.eye(n))
        else:
            self.theta = nn.Parameter(torch.eye(n) if mode == 'hc' else torch.zeros(n, n))

    def mixing_matrix(self):
        if self.mode == 'mhc':
            return sinkhorn(self.theta)
        if self.mode == 'ohc':
            return cayley(self.theta)
        return self.theta

    def forward(self, H, fn):
        x = torch.einsum('blnd,n->bld', H, self.alpha)
        y = fn(self.norm(x))
        A = self.mixing_matrix()
        return torch.einsum('ij,bljd->blid', A, H) + torch.einsum('i,bld->blid', self.beta, y)


class CausalAttn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.heads, D // self.heads).transpose(1, 2)
        k = k.view(B, L, self.heads, D // self.heads).transpose(1, 2)
        v = v.view(B, L, self.heads, D // self.heads).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(B, L, D))


class FFN(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        self.fc1 = nn.Linear(d, mult * d)
        self.fc2 = nn.Linear(mult * d, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class ResBlock(nn.Module):
    """Standard pre-norm transformer block."""

    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.att, self.ffn = CausalAttn(d, heads), FFN(d)

    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.ffn(self.n2(x))


class HCBlock(nn.Module):
    def __init__(self, d, heads, mode, n):
        super().__init__()
        self.att, self.ffn = CausalAttn(d, heads), FFN(d)
        self.hc_att, self.hc_ffn = HCConn(d, n, mode), HCConn(d, n, mode)

    def forward(self, H):
        H = self.hc_att(H, self.att)
        return self.hc_ffn(H, self.ffn)


class TSFM(nn.Module):
    def __init__(self, patch=16, d_model=256, layers=24, heads=8, max_len=4096,
                 hc_mode='res', hc_expand=4):
        super().__init__()
        self.patch = patch
        self.hc_mode = hc_mode
        self.n = hc_expand
        self.use_hc = hc_mode != 'res'
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len // patch, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        if self.use_hc:
            self.blocks = nn.ModuleList([HCBlock(d_model, heads, hc_mode, hc_expand) for _ in range(layers)])
            self.readout = nn.Parameter(torch.zeros(hc_expand))
            self.readout.data[0] = 1.0
        else:
            self.blocks = nn.ModuleList([ResBlock(d_model, heads) for _ in range(layers)])
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, patch)

    def forward(self, x):
        # x: [B, T] raw series -> instance norm
        mu, sd = x.mean(1, keepdim=True), x.std(1, keepdim=True) + 1e-5
        x = (x - mu) / sd
        L = x.shape[1] // self.patch
        p = x[:, :L * self.patch].unfold(1, self.patch, self.patch)  # [B, L, patch]
        z = self.embed(p) + self.pos[:, :L]
        if self.use_hc:
            H = z.unsqueeze(2).expand(-1, -1, self.n, -1).contiguous()
            for blk in self.blocks:
                H = blk(H)
            z = torch.einsum('blnd,n->bld', H, self.readout)
        else:
            for blk in self.blocks:
                z = blk(z)
        z = self.final_norm(z)
        pred = self.head(z)  # [B, L, patch]
        # next-patch prediction: pred[:, i] ~ p[:, i+1]
        loss = F.mse_loss(pred[:, :-1], p[:, 1:])
        return loss, z
