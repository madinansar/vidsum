"""Kendall's tau and Spearman's rho between predicted step scores and each annotator's scores.

Follows Otani et al. (CVPR 2019) as implemented in CA-SUM's choose_best_model.py: TVSum user scores are
normalized to [0, 1], sampled at the step positions, correlated per annotator, averaged over annotators,
then over videos. SumMe has no per-frame importance annotations, so it is not supported.
"""
import csv

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr


def load_tvsum_user_scores(tsv_path):
    """{'video_k': [n_users, n_frames] array}, keyed in TSV order (which is the order of the official h5)."""
    order, scores = [], {}
    with open(tsv_path, encoding="utf-8", errors="ignore") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            video_id = row[0]
            user = np.array([float(x) for x in row[2].split(",")])
            user = user / user.max(initial=-1)
            if video_id not in scores:
                order.append(video_id)
                scores[video_id] = []
            scores[video_id].append(user)
    return {f"video_{i}": np.stack(scores[vid]) for i, vid in enumerate(order, start=1)}


def rank_correlation(step_scores, user_frame_scores, picks):
    """Mean (rho, tau) over annotators for one video."""
    pred = np.asarray(step_scores, dtype=np.float64)
    users = user_frame_scores[:, np.asarray(picks, dtype=np.int64)]
    n = min(len(pred), users.shape[1])
    rhos, taus = [], []
    for user in users:
        rhos.append(spearmanr(pred[:n], user[:n])[0])
        taus.append(kendalltau(rankdata(pred[:n]), rankdata(user[:n]))[0])
    return float(np.nanmean(rhos)), float(np.nanmean(taus))
