"""One entry point for all summary metrics: evaluate_scores(dataset, {video_key: step_scores})."""
from functools import lru_cache

import h5py
import numpy as np

from ..paths import dataset_info
from .highlight_map import highlight_map
from .rank_corr import load_tvsum_user_scores, rank_correlation
from .summary import evaluate_summary, generate_summary


@lru_cache(maxsize=None)
def _ground_truth(dataset):
    info = dataset_info(dataset)
    gt = {}
    with h5py.File(info["official_h5"], "r") as hdf:
        for key in hdf.keys():
            g = hdf[key]
            gt[key] = {
                "user_summary": np.asarray(g["user_summary"]),
                "change_points": np.asarray(g["change_points"]),
                "n_frames": int(np.asarray(g["n_frames"])),
                "picks": np.asarray(g["picks"]),
                "gtscore": np.asarray(g["gtscore"], dtype=np.float32),
            }
    return gt


@lru_cache(maxsize=None)
def _user_scores(dataset):
    tsv = dataset_info(dataset)["anno_tsv"]
    return load_tvsum_user_scores(tsv) if tsv is not None else None


def evaluate_scores(dataset, scores, budget_ratio=0.15, per_video=False):
    """Return mean F1 (%), Spearman's rho, Kendall's tau (TVSum only) and mAP@50/15 for the given videos."""
    dataset = dataset.lower()
    gt = _ground_truth(dataset)
    users = _user_scores(dataset)
    eval_method = dataset_info(dataset)["eval_method"]

    rows = {}
    for key, pred in scores.items():
        pred = np.asarray(pred, dtype=np.float32)
        g = gt[key]
        summary = generate_summary(pred, g["change_points"], g["n_frames"], g["picks"], budget_ratio)
        row = {"f1": evaluate_summary(summary, g["user_summary"], eval_method)}
        if users is not None:
            row["rho"], row["tau"] = rank_correlation(pred, users[key], g["picks"])
        rows[key] = row

    result = {name: float(np.mean([r[name] for r in rows.values()])) for name in next(iter(rows.values()))}
    keys = list(scores)
    result["map50"], result["map15"] = highlight_map(
        [np.asarray(scores[k], dtype=np.float32) for k in keys], [gt[k]["gtscore"] for k in keys]
    )
    if per_video:
        result["per_video"] = rows
    return result
