"""Blending must return a true weighted average, not one biased by an epsilon.

process_chunk accumulated a Gaussian-weighted sum and then divided by
`weights_b + epsilon` with `where=weights_b > 0`. The `where=` clause already
excludes every zero-weight voxel - the comment above it says so - and the
Gaussian map's minimum is exp(-3*4^2/2) = 3.8e-11, which is 265x SMALLER than
the 1e-8 epsilon. So wherever a voxel is covered by a single patch and sits far
from that patch's centre, the denominator was dominated by the epsilon and the
result was scaled by w/(w+eps) instead of averaged. At the Gaussian's minimum
that factor is about 0.0038, a 99.6% reduction.

The ground truth needs no model: a weighted average of a constant field must
return that constant at every covered voxel, however small the weights.

test_constant_field_is_preserved_at_the_patch_corner fails against the previous
implementation.
"""

import numpy as np
import pytest
import zarr

from vesuvius.models.run import blending
from vesuvius.models.run.blending import generate_gaussian_map, process_chunk

PATCH = (8, 8, 8)
NUM_CLASSES = 1
VALUE = 1.0


@pytest.fixture
def single_patch_chunk(tmp_path, monkeypatch):
    """One patch at the origin, constant logits, covering the whole chunk."""
    pz, py, px = PATCH

    logits_path = tmp_path / "logits.zarr"
    logits = zarr.open(
        str(logits_path), mode="w", shape=(1, NUM_CLASSES, pz, py, px),
        chunks=(1, NUM_CLASSES, pz, py, px), dtype=np.float32,
    )
    logits[0] = np.full((NUM_CLASSES, pz, py, px), VALUE, dtype=np.float32)

    output_path = tmp_path / "out.zarr"
    output = zarr.open(
        str(output_path), mode="w", shape=(NUM_CLASSES, pz, py, px),
        chunks=(NUM_CLASSES, pz, py, px), dtype=np.float16,
    )

    monkeypatch.setattr(blending, "_worker_state", {
        "part_files": {0: {"logits": str(logits_path)}},
        "gaussian_map": generate_gaussian_map(PATCH),
        "patch_size": PATCH,
        "num_classes": NUM_CLASSES,
        "is_s3": False,
        "logits_stores": {},
        "output_store": output,
        "finalize_config": None,
    }, raising=False)

    chunk_info = {"z_start": 0, "z_end": pz, "y_start": 0, "y_end": py,
                  "x_start": 0, "x_end": px}
    return chunk_info, {0: [(0, 0, 0, 0)]}, output


def test_gaussian_minimum_is_far_below_a_1e_8_epsilon():
    """The magnitude that makes the bias matter."""
    gaussian = generate_gaussian_map(PATCH)
    assert gaussian.max() == pytest.approx(1.0)
    assert gaussian.min() < 1e-8 / 100, (
        f"gaussian min {gaussian.min():.3e} should be far below a 1e-8 epsilon"
    )


def test_constant_field_is_preserved_at_the_patch_corner(single_patch_chunk):
    chunk_info, chunk_patches, output = single_patch_chunk

    result = process_chunk(chunk_info=chunk_info, chunk_patches=chunk_patches)
    assert result["patches_processed"] == 1

    blended = np.asarray(output[:]).astype(np.float32)

    # every covered voxel must come back as the constant that went in
    np.testing.assert_allclose(blended, VALUE, rtol=2e-3)


def test_corner_voxel_specifically(single_patch_chunk):
    """The corner carries the Gaussian's minimum weight - the worst case."""
    chunk_info, chunk_patches, output = single_patch_chunk

    process_chunk(chunk_info=chunk_info, chunk_patches=chunk_patches)
    corner = float(np.asarray(output[:])[0, 0, 0, 0])

    assert corner == pytest.approx(VALUE, rel=2e-3), (
        f"corner came back {corner!r}; an epsilon in the denominator drives it toward 0"
    )
