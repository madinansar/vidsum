"""Unsupervised training losses. CA-SUM itself uses only the length regularizer."""
import torch


def length_regularization(scores, reg_factor):
    """CA-SUM eq. (1): pull the mean importance score towards sigma."""
    return torch.abs(scores.mean() - reg_factor)


def smoothness(scores):
    """Penalize step-to-step jumps in the importance curve."""
    if scores.numel() < 2:
        return scores.new_zeros(())
    return (scores[1:] - scores[:-1]).abs().mean()


def unsupervised_loss(scores, cfg):
    """Weighted sum of the enabled terms; returns (total, {term: value})."""
    terms = {"length": length_regularization(scores, cfg["reg_factor"])}
    if cfg.get("smooth_weight", 0.0) > 0:
        terms["smooth"] = cfg["smooth_weight"] * smoothness(scores)
    total = sum(terms.values())
    return total, {k: float(v.detach()) for k, v in terms.items()}
