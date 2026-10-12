"""Step scores -> keyshot summary -> F1, following CA-SUM's evaluation code."""
import numpy as np

from .knapsack import knapsack


def upsample_scores(step_scores, n_frames, picks):
    """Spread scores of the sub-sampled steps (at `picks`) over every frame of the original video."""
    frame_scores = np.zeros(int(n_frames), dtype=np.float32)
    positions = np.asarray(picks).astype(np.int64)
    if len(positions) == 0 or positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        left, right = int(positions[i]), int(positions[i + 1])
        frame_scores[left:right] = float(step_scores[i]) if i < len(step_scores) else 0.0
    return frame_scores


def generate_summary(step_scores, change_points, n_frames, picks, budget_ratio=0.15):
    """Binary frame-level summary: shots ranked by mean score, picked by knapsack under `budget_ratio`."""
    frame_scores = upsample_scores(step_scores, n_frames, picks)

    shot_lengths, shot_scores = [], []
    for start, end in change_points:
        start, end = int(start), int(end)
        shot_lengths.append(end - start + 1)
        shot_scores.append(float(frame_scores[start:end + 1].mean()))

    total_frames = int(change_points[-1][1]) + 1
    selected = knapsack(int(total_frames * budget_ratio), shot_lengths, shot_scores)

    summary = np.zeros(total_frames, dtype=np.int8)
    for shot in selected:
        start, end = change_points[shot]
        summary[int(start):int(end) + 1] = 1
    return summary


def evaluate_summary(predicted_summary, user_summary, eval_method="avg"):
    """F1 (in %) of a binary summary against every user summary, reduced by 'avg' (TVSum) or 'max' (SumMe)."""
    max_len = max(len(predicted_summary), user_summary.shape[1])
    pred = np.zeros(max_len, dtype=int)
    gt = np.zeros(max_len, dtype=int)
    pred[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        gt[:user_summary.shape[1]] = user_summary[user]
        overlap = int((pred & gt).sum())
        if pred.sum() == 0 or gt.sum() == 0 or overlap == 0:
            f_scores.append(0.0)
            continue
        precision = overlap / pred.sum()
        recall = overlap / gt.sum()
        f_scores.append(2 * precision * recall * 100 / (precision + recall))

    if eval_method == "max":
        return float(max(f_scores))
    return float(sum(f_scores) / len(f_scores))
