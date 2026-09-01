"""Every blend weight must clear the guard the accumulators normalize with.

The Hann branch of compute_importance_map_2d floors at 0.001 with a comment
saying why: "An exact Hann window is zero at its outermost pixels; flooring
keeps a boundary covered by only one patch normalizable." The Gaussian branch
floored at float32 eps = 1.19e-7 instead, and the native path's
create_importance_map did the same.

Both accumulators normalize with `where=weight > 1e-6` - infer.py in the flat
writer and ChunkAccumulator3D._flush in the native one. 1.19e-7 is below that,
so where the guard fires the accumulator keeps its un-normalized weighted sum
(p*w, around 1e-7), which clips and truncates to a uint8 0. A probability of
0.80 was written out as 0 - not "unknown", not masked, but confident
background.

The gaussian cases fail against the previous implementation.
"""

import numpy as np
import pytest
import torch

from vesuvius.ink_detection.inference.infer import compute_importance_map_2d
from vesuvius.ink_detection.inference.infer_full3d_tifxyz import create_importance_map

# The threshold both accumulators use when dividing by the accumulated weight.
NORMALIZER_GUARD = 1e-6


def test_the_floor_clears_the_normalizer_guard():
    # Imported in-test so the behavioral tests below still collect and run
    # against a tree without the constant.
    from vesuvius.ink_detection.inference.inference_runtime import MIN_BLEND_WEIGHT

    assert MIN_BLEND_WEIGHT > NORMALIZER_GUARD


@pytest.mark.parametrize("mode", ["constant", "hann", "gaussian"])
@pytest.mark.parametrize("patch", [(64, 64), (128, 96)])
def test_flat_weights_are_normalizable_everywhere(mode, patch):
    weight = compute_importance_map_2d(patch_size=patch, mode=mode)

    assert float(weight.min()) > NORMALIZER_GUARD, (
        f"{mode} map has weights the normalizer skips, so those pixels keep an "
        f"un-normalized sum and truncate to 0"
    )
    assert float(weight.max()) == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["constant", "gaussian"])
def test_native_weights_are_normalizable_everywhere(mode):
    weight = create_importance_map((16, 16, 16), mode=mode)

    assert float(np.min(weight)) > NORMALIZER_GUARD
    assert float(np.max(weight)) == pytest.approx(1.0)


def test_a_confident_prediction_at_the_patch_edge_survives_normalization():
    """The consequence, end to end through the same arithmetic."""
    weight = compute_importance_map_2d(patch_size=(64, 64), mode="gaussian")
    probability = torch.full_like(weight, 0.8)

    accumulated = (probability * weight).numpy().astype(np.float32)
    accumulated_weight = weight.numpy().astype(np.float32)

    np.divide(
        accumulated, accumulated_weight, out=accumulated,
        where=accumulated_weight > NORMALIZER_GUARD,
    )
    encoded = np.clip(accumulated, 0, 1) * 255

    assert encoded.min() > 0, "a confident 0.8 was written out as background"
    np.testing.assert_allclose(encoded, 0.8 * 255, rtol=1e-4)
