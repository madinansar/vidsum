# -*- coding: utf-8 -*-
import numpy as np


def calc_scatters(kernel):
    """Scatter matrix used by kernel temporal segmentation DP."""
    n = kernel.shape[0]
    diag_csum = np.cumsum([0.0] + list(np.diag(kernel)))
    k2 = np.zeros((n + 1, n + 1), dtype=np.float64)
    k2[1:, 1:] = np.cumsum(np.cumsum(kernel, axis=0), axis=1)

    diag_k2 = np.diag(k2)
    i = np.arange(n).reshape((-1, 1))
    j = np.arange(n).reshape((1, -1))

    denom = (j - i + 1).astype(np.float64) + (j == i - 1).astype(np.float64)
    scatters = (
        diag_csum[1:].reshape((1, -1))
        - diag_csum[:-1].reshape((-1, 1))
        - (
            diag_k2[1:].reshape((1, -1))
            + diag_k2[:-1].reshape((-1, 1))
            - k2[1:, :-1].T
            - k2[:-1, 1:]
        )
        / np.clip(denom, 1e-12, None)
    )
    scatters[j < i] = 0.0
    return scatters


def cpd_nonlin(kernel, ncp, lmin=1, lmax=100000, backtrack=True):
    """Kernel change point detection with dynamic programming."""
    m = int(ncp)
    n = kernel.shape[0]

    if n < (m + 1) * lmin:
        m = max(0, (n // max(lmin, 1)) - 1)

    scatters = calc_scatters(kernel)

    costs = 1e101 * np.ones((m + 1, n + 1), dtype=np.float64)
    costs[0, lmin : min(lmax, n) + 1] = scatters[0, lmin - 1 : min(lmax, n) - 1 + 1]

    if backtrack:
        prev = np.zeros((m + 1, n + 1), dtype=np.int64)
    else:
        prev = np.zeros((1, 1), dtype=np.int64)

    for k in range(1, m + 1):
        for l in range((k + 1) * lmin, n + 1):
            tmin = max(k * lmin, l - lmax)
            tmax = l - lmin + 1
            c = scatters[tmin:tmax, l - 1].reshape(-1) + costs[k - 1, tmin:tmax].reshape(-1)
            best = int(np.argmin(c))
            costs[k, l] = c[best]
            if backtrack:
                prev[k, l] = best + tmin

    cps = np.zeros(m, dtype=np.int64)
    if backtrack and m > 0:
        cur = n
        for k in range(m, 0, -1):
            cps[k - 1] = prev[k, cur]
            cur = cps[k - 1]

    scores = costs[:, n].copy()
    scores[scores > 1e99] = np.inf
    return cps, scores


def cpd_auto(kernel, ncp_max, vmax, desc_rate=1, lmin=1, lmax=100000):
    """Automatically pick number of KTS change points with penalty."""
    m = int(max(0, ncp_max))
    if m == 0:
        return np.array([], dtype=np.int64)

    _, raw_scores = cpd_nonlin(kernel, m, lmin=lmin, lmax=lmax, backtrack=False)

    n = kernel.shape[0]
    n2 = n * max(1, int(desc_rate))

    penalties = np.zeros(m + 1, dtype=np.float64)
    ncp = np.arange(1, m + 1)
    penalties[1:] = (vmax * ncp / (2.0 * n2)) * (np.log(float(n2) / np.clip(ncp, 1, None)) + 1.0)

    costs = raw_scores / float(max(n, 1)) + penalties
    m_best = int(np.argmin(costs))
    cps, _ = cpd_nonlin(kernel, m_best, lmin=lmin, lmax=lmax, backtrack=True)
    return cps


def kts_segments_from_features(features, max_change_points=20, min_seg_len=1):
    """Compute segment boundaries [b0=0, b1, ..., bN=T] from features [T, D]."""
    t = int(features.shape[0])
    if t <= max(2, min_seg_len):
        return [0, t]

    x = features.astype(np.float64, copy=False)
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    kernel = np.matmul(x, x.T)

    # Heuristic from KTS literature; larger trace means more within-segment variance.
    centered = kernel - kernel.mean(axis=0, keepdims=True) - kernel.mean(axis=1, keepdims=True) + kernel.mean()
    vmax = float(np.trace(centered) / float(max(t, 1)))

    max_cp = int(min(max_change_points, max(0, (t // max(min_seg_len, 1)) - 1)))
    cps = cpd_auto(kernel, ncp_max=max_cp, vmax=max(vmax, 1e-6), lmin=min_seg_len)

    boundaries = [0]
    for cp in cps.tolist():
        cp_i = int(cp)
        if 0 < cp_i < t and cp_i not in boundaries:
            boundaries.append(cp_i)
    boundaries.append(t)
    boundaries = sorted(set(boundaries))

    if boundaries[0] != 0:
        boundaries = [0] + boundaries
    if boundaries[-1] != t:
        boundaries.append(t)

    return boundaries
