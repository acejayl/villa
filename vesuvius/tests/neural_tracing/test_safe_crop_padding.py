"""safe_crop_with_padding must always return a crop_size cube.

Its docstring promises "Cropped tensor with shape [batch, crop_size,
crop_size, crop_size, ...]". The implementation clamped only one side of each
bound:

    actual_min = max(min_corner, 0)
    actual_max = min(min_corner + crop_size, spatial_shape)

so when the requested window lies entirely below the volume, actual_max stays
negative and is used directly as a Python slice endpoint. tensor[0:-4] does not
select an empty slice, it selects shape-4 real voxels from the wrong place. The
result was both the wrong shape and the wrong contents. When the window lies
entirely above, the slice is empty but pad_after overshoots and the result is
again too deep.

Both are silent for batch size 1. For batch > 1 with mixed corners the closing
torch.stack raises "stack expects each tensor to be equal size".

These tests fail against the previous implementation.
"""

import pytest
import torch

from vesuvius.neural_tracing.heatmap_single_point.cropping import safe_crop_with_padding

D = 40
CROP = 16


def _volume(batch=1):
    vol = torch.arange(1, 1 + D * D * D, dtype=torch.float32).reshape(1, D, D, D)
    return vol.expand(batch, D, D, D) if batch > 1 else vol


def _crop(volume, corner):
    return safe_crop_with_padding(
        volume, torch.tensor([corner], dtype=torch.float32), CROP
    )


@pytest.mark.parametrize(
    "corner,expected_planes",
    [
        ([8, 8, 8], 16),    # fully inside
        ([-4, 8, 8], 12),   # straddles the low face
        ([36, 8, 8], 4),    # straddles the high face
        ([-20, 8, 8], 0),   # entirely below the volume
        ([60, 8, 8], 0),    # entirely above the volume
    ],
    ids=["inside", "partial_low", "partial_high", "wholly_below", "wholly_above"],
)
def test_shape_and_real_voxel_count(corner, expected_planes):
    out = _crop(_volume(), corner)

    assert tuple(out.shape) == (1, CROP, CROP, CROP)
    assert int((out != 0).sum()) == expected_planes * CROP * CROP


def test_out_of_bounds_window_is_all_zeros_not_misplaced_data():
    """The failure mode was real voxels appearing in a window with no overlap."""
    out = _crop(_volume(), [-20, 8, 8])
    assert bool((out == 0).all()), "real voxel data leaked into a disjoint window"


def test_padding_is_placed_on_the_correct_side():
    volume = _volume()
    out = _crop(volume, [-4, 0, 0])

    assert bool((out[0, :4] == 0).all()), "padding not at the low end"
    assert torch.equal(out[0, 4], volume[0, 0, :CROP, :CROP])


def test_batched_mixed_bounds_stacks():
    corners = torch.tensor([[8, 8, 8], [-20, 8, 8]], dtype=torch.float32)
    out = safe_crop_with_padding(_volume(batch=2), corners, CROP)

    assert tuple(out.shape) == (2, CROP, CROP, CROP)
    assert bool((out[1] == 0).all())


def test_trailing_dims_are_preserved():
    volume = torch.ones(1, D, D, D, 3)
    out = _crop(volume, [-20, 8, 8])
    assert tuple(out.shape) == (1, CROP, CROP, CROP, 3)
