"""The ported CA-SUM must match upstream CA-SUM (git HEAD of ../CA-SUM, not the locally edited files)."""
import subprocess
import sys

import pytest
import torch

from vidsum.models.ca_sum import CASUM, MultimodalCASUM, init_weights
from vidsum.paths import ROOT

UPSTREAM = ROOT.parent / "CA-SUM"


def _git_show(path):
    return subprocess.run(["git", "-C", str(UPSTREAM), "show", f"HEAD:{path}"],
                          capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def upstream_casum(tmp_path_factory):
    if not (UPSTREAM / ".git").exists():
        pytest.skip("upstream CA-SUM repo not found")
    root = tmp_path_factory.mktemp("upstream")
    (root / "layers").mkdir()
    (root / "layers" / "__init__.py").write_text("")
    for name in ("attention", "summarizer"):
        (root / "layers" / f"{name}.py").write_text(_git_show(f"model/layers/{name}.py"))
    sys.path.insert(0, str(root))
    try:
        from layers.summarizer import CA_SUM
    finally:
        sys.path.remove(str(root))
    return CA_SUM


@pytest.mark.parametrize("length", [75, 233])
def test_forward_matches_upstream(upstream_casum, length):
    torch.manual_seed(0)
    ref = upstream_casum(input_size=1024, output_size=1024, block_size=60).eval()
    ours = CASUM(1024, 1024, 60).eval()
    ours.load_state_dict(ref.state_dict())

    x = torch.randn(length, 1024)
    with torch.no_grad():
        y_ref, a_ref = ref(x)
        y_ours, a_ours = ours(x)
    assert torch.allclose(y_ref, y_ours, atol=1e-6)
    assert torch.allclose(a_ref, a_ours, atol=1e-6)


def test_single_modality_wrapper_is_plain_casum_with_same_init(upstream_casum):
    """Same parameter names and order -> same seed gives identical initial weights as upstream."""
    torch.manual_seed(12345)
    ref = upstream_casum(input_size=1024, output_size=1024, block_size=60)
    init_weights(ref)  # same loop as CA-SUM's Solver.init_weights
    torch.manual_seed(12345)
    ours = MultimodalCASUM({"googlenet": 1024})
    init_weights(ours)
    ref_state = ref.state_dict()
    for k, v in ours.summarizer.state_dict().items():
        assert torch.equal(v, ref_state[k]), k


def test_short_video_has_no_nan():
    model = CASUM(64, 64, 60).eval()
    with torch.no_grad():
        y, _ = model(torch.randn(40, 64))
    assert torch.isfinite(y).all()
