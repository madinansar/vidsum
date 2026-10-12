"""CA-SUM summarizer and its multimodal extension."""
import numpy as np
import torch.nn as nn

from .ca_attention import ConcentratedAttention
from .fusion import build_fusion


class CASUM(nn.Module):
    """Ported from CA-SUM/model/layers/summarizer.py: attention + residual + 2-layer regressor -> [1, T] scores."""

    def __init__(self, input_size=1024, output_size=1024, block_size=60):
        super().__init__()
        self.attention = ConcentratedAttention(input_size, output_size, block_size)
        self.linear_1 = nn.Linear(input_size, input_size)
        self.linear_2 = nn.Linear(input_size, 1)
        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(input_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(input_size, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, frame_features):
        weighted_value, attn_weights = self.attention(frame_features)
        y = self.norm_y(self.drop(frame_features + weighted_value))
        y = self.norm_linear(self.drop(self.relu(self.linear_1(y))))
        y = self.sigmoid(self.linear_2(y))
        return y.view(1, -1), attn_weights


class MultimodalCASUM(nn.Module):
    """Per-modality projection -> fusion -> CA-SUM.

    With a single modality and `d_model=None` there is no projection, so the model is exactly CA-SUM.
    """

    def __init__(self, modality_dims, d_model=None, fusion="concat", block_size=60, heads=8):
        super().__init__()
        self.modalities = list(modality_dims)
        if d_model is None:
            if len(self.modalities) != 1:
                raise ValueError("d_model is required when fusing more than one modality")
            d_model = modality_dims[self.modalities[0]]
            self.proj = nn.ModuleDict({m: nn.Identity() for m in self.modalities})
        else:
            self.proj = nn.ModuleDict({
                m: nn.Sequential(nn.Linear(dim, d_model), nn.LayerNorm(d_model)) for m, dim in modality_dims.items()
            })
        self.d_model = d_model
        self.fusion = build_fusion(fusion, len(self.modalities), d_model, heads=heads)
        self.summarizer = CASUM(d_model, d_model, block_size)

    def fuse(self, features, masks=None, gates=None):
        """features/masks: {modality: [T, D] / [T]}; gates: [T, M] or None. Returns fused [T, d_model]."""
        feats = []
        for m in self.modalities:
            f = self.proj[m](features[m])
            if masks is not None:
                f = f * masks[m].unsqueeze(-1)
            feats.append(f)
        return self.fusion(feats, gates)

    def forward(self, features, masks=None, gates=None):
        """Returns (scores [T], attention [T, T], fused features [T, d_model])."""
        fused = self.fuse(features, masks, gates)
        scores, attn = self.summarizer(fused)
        return scores.squeeze(0), attn, fused


def init_weights(net, init_type="xavier", init_gain=1.4142):
    """CA-SUM's initialization (solver.py): weights by `init_type`, biases 0.1, LayerNorms untouched."""
    for name, param in net.named_parameters():
        if "weight" in name and "norm" not in name and param.dim() > 1:
            if init_type == "normal":
                nn.init.normal_(param, mean=0.0, std=init_gain)
            elif init_type == "xavier":
                nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(param, gain=np.sqrt(2.0))
            else:
                raise NotImplementedError(f"initialization method {init_type} is not implemented.")
        elif "bias" in name:
            nn.init.constant_(param, 0.1)
