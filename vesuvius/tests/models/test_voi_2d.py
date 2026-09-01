"""VOIMetric must work for 2D runs, not only 3D.

BaseTrainer._initialize_evaluation_metrics attaches VOIMetric to every target
regardless of dimensionality, and the training loop calls metric.update() with
no try/except. For a 2D run:

  * _bbox3d asserted mask.ndim == 3 and died with a bare AssertionError, and
  * the gt reshaping had no arm for (batch, channels, height, width) with more
    than one channel, so a multi-channel 2D target was never reduced and the
    rank check raised ValueError.

Both shipped 2D segmentation configs hit one of these at the first validation
epoch: single_task/ps256_medial_2d.yaml (2 channels) and
single_task/ps256_normals.yaml (3 channels). docs/training_flow.md documents 2D
training as supported.

The 2D tests fail against the previous implementation; the 3D ones pass before
and after and guard the path that already worked.
"""

import numpy as np
import pytest
import torch

from vesuvius.models.evaluation.voi import compute_voi

RNG = np.random.default_rng(0)


def _mask(*shape):
    return torch.tensor(RNG.integers(0, 2, shape).astype(np.float32))


def _assert_scores(result):
    assert set(result) >= {"voi_total", "voi_split", "voi_merge", "voi_score"}
    assert np.isfinite(result["voi_total"])
    assert 0.0 <= result["voi_score"] <= 1.0


def test_2d_single_channel_target():
    """ps256_medial_2d-shaped call: pred (B,H,W), gt (B,1,H,W)."""
    result = compute_voi(pred=_mask(2, 16, 16), gt=_mask(2, 1, 16, 16))
    _assert_scores(result)


@pytest.mark.parametrize("channels", [2, 3], ids=["medial_2d", "normals"])
def test_2d_multi_channel_target(channels):
    """Both shipped 2D configs: a channel axis that must be argmaxed away."""
    result = compute_voi(pred=_mask(2, channels, 16, 16), gt=_mask(2, channels, 16, 16))
    _assert_scores(result)


def test_3d_still_works():
    result = compute_voi(pred=_mask(2, 2, 8, 8, 8), gt=_mask(2, 2, 8, 8, 8))
    _assert_scores(result)


def test_bbox_handles_2d_and_3d():
    # imported here so the behavioural tests above still collect and run
    # against a tree that does not have this helper yet
    from vesuvius.models.evaluation.voi import _bbox_nd

    m2 = np.zeros((8, 8), bool)
    m2[2:5, 3:7] = True
    assert _bbox_nd(m2) == (slice(2, 5), slice(3, 7))

    m3 = np.zeros((6, 6, 6), bool)
    m3[1:3, 2:5, 0:4] = True
    assert _bbox_nd(m3) == (slice(1, 3), slice(2, 5), slice(0, 4))

    assert _bbox_nd(np.zeros((4, 4), bool)) is None


def test_identical_masks_score_perfectly_in_2d():
    """A sanity anchor: prediction == ground truth must be a perfect score."""
    m = np.zeros((1, 1, 12, 12), np.float32)
    m[0, 0, 2:8, 3:9] = 1.0
    t = torch.tensor(m)
    result = compute_voi(pred=t.squeeze(1), gt=t)
    assert result["voi_total"] == pytest.approx(0.0, abs=1e-9)
    assert result["voi_score"] == pytest.approx(1.0, abs=1e-9)
