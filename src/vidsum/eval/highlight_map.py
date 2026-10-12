"""mAP@{50,15} highlight-detection metric (as used by TripleSumm / Mr.HiSum), computed on step-level scores."""
import math

import numpy as np
from sklearn.metrics import average_precision_score


def _average_precision(pred, gt, rho, shot_len):
    num_shots = int(math.ceil(len(pred) / shot_len))
    pad = num_shots * shot_len - len(pred)
    pred_shots = np.pad(pred, (0, pad)).reshape(-1, shot_len).mean(axis=1)
    gt_shots = np.pad(gt, (0, pad)).reshape(-1, shot_len).mean(axis=1)

    top_k = int(math.ceil(num_shots * rho))
    labels = np.zeros(num_shots)
    labels[np.argsort(gt_shots)[-top_k:]] = 1
    return float(average_precision_score(labels, pred_shots))


def highlight_map(pred_list, gt_list, shot_len=5):
    """Mean AP (in %) at top-50% and top-15% ground-truth shots. `shot_len` is in steps."""
    ap50, ap15 = [], []
    for pred, gt in zip(pred_list, gt_list):
        n = min(len(pred), len(gt))
        pred, gt = np.asarray(pred[:n], dtype=np.float64), np.asarray(gt[:n], dtype=np.float64)
        ap50.append(_average_precision(pred, gt, 0.50, shot_len))
        ap15.append(_average_precision(pred, gt, 0.15, shot_len))
    return float(np.mean(ap50)) * 100.0, float(np.mean(ap15)) * 100.0
