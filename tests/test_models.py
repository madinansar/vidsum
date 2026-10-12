import pytest
import torch

from vidsum.models.ca_sum import MultimodalCASUM, init_weights
from vidsum.models.gating_agent import ModalityGatePolicy

DIMS = {"googlenet": 1024, "audio_pann": 2048, "text": 384}


def _inputs(T=130):
    feats = {m: torch.randn(T, d) for m, d in DIMS.items()}
    masks = {m: torch.ones(T) for m in DIMS}
    return feats, masks


@pytest.mark.parametrize("fusion", ["concat", "sum", "xattn", "softgate"])
def test_fusions_forward_backward(fusion):
    model = MultimodalCASUM(DIMS, d_model=128, fusion=fusion)
    init_weights(model)
    feats, masks = _inputs()
    scores, attn, fused = model(feats, masks, gates=torch.ones(130, 3))
    assert scores.shape == (130,) and attn.shape == (130, 130) and fused.shape == (130, 128)
    scores.mean().backward()


def test_gate_policy_drives_fusion():
    model = MultimodalCASUM(DIMS, d_model=64, fusion="concat")
    policy = ModalityGatePolicy(num_modalities=3, dim=64)
    feats, masks = _inputs()
    projected = [model.proj[m](feats[m]) for m in model.modalities]
    boundaries = [0, 40, 90, 130]
    gates, log_prob, probs = policy.act(projected, boundaries)
    assert gates.shape == (130, 3) and probs.shape == (3, 3)
    scores, _, _ = model(feats, masks, gates=gates)
    (-scores.mean().detach() * log_prob).backward()
    assert policy.net[0].weight.grad is not None
