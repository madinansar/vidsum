"""Fusion of per-modality step features into one [T, d] sequence for the concentrated attention.

All fusions take `gates`: a [T, M] tensor in [0, 1] (or None = all ones) that scales each modality's
contribution per step. Static baselines leave it None; the RL modality-gating agent supplies a sampled
binary mask, so every fusion here can be gated without changes.
"""
import torch
import torch.nn as nn


def _split_gates(gates, num_modalities, length, device):
    if gates is None:
        return [torch.ones(length, 1, device=device)] * num_modalities
    return [gates[:, i:i + 1] for i in range(num_modalities)]


class IdentityFusion(nn.Module):
    """Single modality: pass through (with the CA-SUM baseline this keeps the original 1024-d input)."""

    def forward(self, feats, gates=None):
        return feats[0] if gates is None else feats[0] * gates[:, :1]


class ConcatFusion(nn.Module):
    def __init__(self, num_modalities, dim):
        super().__init__()
        self.proj = nn.Linear(num_modalities * dim, dim)

    def forward(self, feats, gates=None):
        g = _split_gates(gates, len(feats), feats[0].shape[0], feats[0].device)
        return self.proj(torch.cat([f * gi for f, gi in zip(feats, g)], dim=-1))


class SumFusion(nn.Module):
    def forward(self, feats, gates=None):
        g = _split_gates(gates, len(feats), feats[0].shape[0], feats[0].device)
        return sum(f * gi for f, gi in zip(feats, g))


class CrossAttentionFusion(nn.Module):
    """The first modality (visual) queries every other modality; contexts are added residually."""

    def __init__(self, num_modalities, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True) for _ in range(num_modalities - 1)
        )

    def forward(self, feats, gates=None):
        g = _split_gates(gates, len(feats), feats[0].shape[0], feats[0].device)
        query = feats[0].unsqueeze(0)
        out = feats[0] * g[0]
        for attn, other, gi in zip(self.attn, feats[1:], g[1:]):
            ctx, _ = attn(query, other.unsqueeze(0), other.unsqueeze(0), need_weights=False)
            out = out + ctx.squeeze(0) * gi
        return out


class SoftGateFusion(nn.Module):
    """Learned (non-RL) per-step modality weights: the differentiable baseline the RL gate is compared to."""

    def __init__(self, num_modalities, dim):
        super().__init__()
        self.gate = nn.Linear(num_modalities * dim, num_modalities)
        self.last_weights = None

    def forward(self, feats, gates=None):
        weights = torch.sigmoid(self.gate(torch.cat(feats, dim=-1)))
        if gates is not None:
            weights = weights * gates
        self.last_weights = weights.detach()
        return sum(f * weights[:, i:i + 1] for i, f in enumerate(feats))


def build_fusion(name, num_modalities, dim, heads=8):
    if num_modalities == 1:
        return IdentityFusion()
    if name == "concat":
        return ConcatFusion(num_modalities, dim)
    if name == "sum":
        return SumFusion()
    if name == "xattn":
        return CrossAttentionFusion(num_modalities, dim, heads=heads)
    if name == "softgate":
        return SoftGateFusion(num_modalities, dim)
    raise ValueError(f"Unknown fusion {name!r}")
