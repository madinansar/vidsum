"""Policy network for RL modality gating (Phase 4).

Decision step: one KTS segment. State: mean-pooled projected features of every modality in the segment.
Action: an independent Bernoulli "on/off" per modality. The sampled segment mask is expanded to a
per-step [T, M] gate that any fusion in fusion.py accepts. Training (REINFORCE with a baseline, reward
from objectives/rewards.py) is wired up in train.py once Phase 3 baselines are in place.
"""
import torch
import torch.nn as nn


class ModalityGatePolicy(nn.Module):
    def __init__(self, num_modalities, dim, hidden=128, init_open_logit=2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_modalities * dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_modalities),
        )
        # Start with every modality mostly "on" so early training matches static fusion.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_open_logit)

    @staticmethod
    def pool_segments(feats, boundaries):
        """feats: list of [T, d] per modality -> [S, M*d] segment states."""
        stacked = torch.cat(feats, dim=-1)
        return torch.stack([stacked[s:e].mean(0) for s, e in zip(boundaries[:-1], boundaries[1:])])

    def forward(self, states):
        """states: [S, M*d] -> Bernoulli probabilities [S, M]."""
        return torch.sigmoid(self.net(states))

    def act(self, feats, boundaries, greedy=False):
        """Sample (or threshold) a gate. Returns (step gates [T, M], log-prob of the action, probs [S, M])."""
        probs = self(self.pool_segments(feats, boundaries))
        if greedy:
            action = (probs > 0.5).float()
            log_prob = torch.zeros((), device=probs.device)
        else:
            dist = torch.distributions.Bernoulli(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
        lengths = torch.tensor([e - s for s, e in zip(boundaries[:-1], boundaries[1:])], device=probs.device)
        step_gates = torch.repeat_interleave(action, lengths, dim=0)
        return step_gates, log_prob, probs
