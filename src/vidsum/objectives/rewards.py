"""Label-free summary rewards for the RL modality-gating agent (Phase 4).

Compute rewards in a FIXED reference space (e.g. L2-normalized raw features of all modalities), not in
the fused space the agent controls; otherwise the agent can raise the reward by changing the space.
"""
import torch
import torch.nn.functional as F


def diversity_reward(selected):
    """1 - mean pairwise cosine similarity among selected steps."""
    if selected.shape[0] < 2:
        return selected.new_zeros(())
    x = F.normalize(selected, dim=-1)
    sim = x @ x.t()
    off_diag = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return 1.0 - sim[off_diag].mean()


def representativeness_reward(selected, all_steps):
    """Mean over all steps of the cosine similarity to their closest selected step."""
    if selected.shape[0] < 1:
        return all_steps.new_zeros(())
    sim = F.normalize(all_steps, dim=-1) @ F.normalize(selected, dim=-1).t()
    return sim.max(dim=-1).values.mean()


def summary_reward(selected, all_steps, w_div=1.0, w_rep=1.0):
    r_div = diversity_reward(selected)
    r_rep = representativeness_reward(selected, all_steps)
    return w_div * r_div + w_rep * r_rep, {"diversity": float(r_div), "representativeness": float(r_rep)}


def modality_cost(step_gates, costs=None):
    """Mean fraction of modalities kept on, optionally weighted by per-modality compute cost [M]."""
    if costs is None:
        return step_gates.mean()
    return (step_gates * costs).sum(-1).mean() / costs.sum()
