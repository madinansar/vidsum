"""Concentrated attention block of CA-SUM (Apostolidis et al., ICMR 2022).

Ported from CA-SUM/model/layers/attention.py. The only change is a clamp on the non-local normalizer,
which is 0/0 (NaN) upstream when a video has fewer steps than one block; outputs are otherwise identical.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcentratedAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, block_size=60):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.block_size = block_size
        self.Wk = nn.Linear(input_size, output_size, bias=False)
        self.Wq = nn.Linear(input_size, output_size, bias=False)
        self.Wv = nn.Linear(input_size, output_size, bias=False)
        self.out = nn.Linear(output_size + 2, input_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_entropy(logits):
        """Row-wise entropy of the attention matrix, normalized by its maximum log(T)."""
        entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        entropy = -1.0 * entropy.sum(-1)
        return entropy / np.log(logits.shape[0])

    def forward(self, x):
        """x: [T, input_size] -> (y: [T, input_size], block-diagonal attention: [T, T])."""
        x_unit = F.normalize(x, p=2, dim=1)
        diversity = 1 - x_unit @ x_unit.t()

        K, Q, V = self.Wk(x), self.Wq(x), self.Wv(x)
        energies = Q @ K.t()
        att_weights = self.softmax(energies)

        entropy = F.normalize(self.get_entropy(energies), p=1, dim=-1)

        D = self.block_size
        num_blocks = math.ceil(energies.shape[0] / D)
        keeping_mask = torch.ones(num_blocks, D, D, device=att_weights.device)
        keeping_mask = torch.block_diag(*keeping_mask)[:att_weights.shape[0], :att_weights.shape[0]]
        zeroing_mask = 1 - keeping_mask
        att_win = att_weights * keeping_mask

        # Non-local dependencies: attention to frames outside the block, weighted by their diversity
        attn_remainder = att_weights * zeroing_mask
        div_remainder = diversity * zeroing_mask
        dep_factor = (div_remainder * attn_remainder).sum(-1).div(div_remainder.sum(-1).clamp_min(1e-12))
        dep_factor = dep_factor.unsqueeze(0).expand(dep_factor.shape[0], -1)
        att_win = att_win + dep_factor * keeping_mask

        y = att_win @ V
        characteristics = torch.stack((entropy, dep_factor[0, :])).detach()
        y = self.out(torch.cat((y, characteristics.t()), dim=-1))
        return y, att_win.clone()
