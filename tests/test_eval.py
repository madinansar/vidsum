import itertools

import numpy as np
import pytest

from vidsum.eval.knapsack import knapsack
from vidsum.eval.summary import evaluate_summary
from vidsum.paths import dataset_info


def test_knapsack_is_optimal():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 7
        weights = rng.integers(1, 6, n).tolist()
        values = rng.random(n).tolist()
        cap = int(rng.integers(3, 15))
        chosen = knapsack(cap, weights, values)
        assert sum(weights[i] for i in chosen) <= cap
        best = max(
            sum(values[i] for i in combo)
            for r in range(n + 1) for combo in itertools.combinations(range(n), r)
            if sum(weights[i] for i in combo) <= cap
        )
        assert np.isclose(sum(values[i] for i in chosen), best)


def test_f1_extremes():
    users = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
    assert evaluate_summary(np.array([1, 1, 0, 0]), users, "avg") == pytest.approx(100.0)
    assert evaluate_summary(np.array([0, 0, 1, 1]), users, "avg") == pytest.approx(0.0)
    assert evaluate_summary(np.array([0, 0, 0, 0]), users, "max") == pytest.approx(0.0)


needs_tvsum = pytest.mark.skipif(not dataset_info("tvsum")["official_h5"].exists(), reason="TVSum data missing")


@needs_tvsum
def test_tvsum_reference_points():
    """Random scores land in the known ~55-58 F1 band with tau ~ 0; ground-truth scores do much better."""
    import h5py

    from vidsum.eval.metrics import evaluate_scores

    rng = np.random.default_rng(0)
    with h5py.File(dataset_info("tvsum")["official_h5"], "r") as hdf:
        keys = list(hdf.keys())
        gt = {k: np.asarray(hdf[f"{k}/gtscore"]) for k in keys}
    random_scores = {k: rng.random(len(v)) for k, v in gt.items()}

    rand = evaluate_scores("tvsum", random_scores)
    oracle = evaluate_scores("tvsum", gt)
    assert 50.0 < rand["f1"] < 62.0
    assert abs(rand["tau"]) < 0.03 and abs(rand["rho"]) < 0.04
    assert oracle["f1"] > rand["f1"] + 3 and oracle["tau"] > 0.3


@needs_tvsum
def test_googlenet_loader_matches_official_steps():
    from vidsum.data.dataset import VideoSumDataset, load_split

    train_keys, test_keys = load_split("tvsum", 0)
    assert len(train_keys) == 40 and len(test_keys) == 10 and not set(train_keys) & set(test_keys)
    ds = VideoSumDataset("tvsum", test_keys[:2], ["googlenet"])
    assert ds.dims == {"googlenet": 1024}
