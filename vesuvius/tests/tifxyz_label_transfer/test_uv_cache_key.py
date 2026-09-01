"""The UV cache key must change when the validity mask changes.

transfer_array's cache_meta recorded the stored shapes, a strided coordinate
fingerprint, the affine, max_distance and nearest_vertices - but never
Surface.valid. The UV field it caches is computed from valid on both sides:
SurfaceMapper indexes only source.valid vertices, locate accepts a quad only
when all four corners are valid, and build_target_uv_map iterates
np.flatnonzero(target.valid). Surface.valid comes from mask.tif and is switched
off wholesale by the documented --ignore-tifxyz-mask flag.

So changing the mask changed the answer but not the key, and the mapping phase
was skipped. docs/reference.md promises the opposite: "a cache whose recorded
configuration does not match is recomputed and rewritten, never silently
reused."

test_validity_mask_changes_the_fingerprint fails against the previous
implementation, which ignored the mask entirely.
"""

import numpy as np
import pytest

from vesuvius.tifxyz_label_transfer.core import _surface_coordinate_fingerprint

H, W = 96, 96


@pytest.fixture
def coords():
    rng = np.random.default_rng(0)
    return tuple(rng.random((H, W)).astype(np.float32) for _ in range(3))


def _fp(coords, valid):
    x, y, z = coords
    return _surface_coordinate_fingerprint(x, y, z, valid)


def test_validity_mask_changes_the_fingerprint(coords):
    full = np.ones((H, W), dtype=bool)

    repaired = full.copy()
    repaired[10:20, 10:20] = False  # the shape a mask repair takes

    assert _fp(coords, full) != _fp(coords, repaired)


def test_ignoring_the_mask_changes_the_fingerprint(coords):
    """--ignore-tifxyz-mask turns every vertex valid; that is a different surface."""
    masked = np.ones((H, W), dtype=bool)
    masked[:, :20] = False

    assert _fp(coords, masked) != _fp(coords, np.ones((H, W), dtype=bool))


def test_a_single_flipped_vertex_changes_the_fingerprint(coords):
    """The popcount term catches changes too small to land on a stride."""
    a = np.ones((H, W), dtype=bool)
    b = a.copy()
    b[H // 2 + 1, W // 2 + 1] = False

    assert _fp(coords, a) != _fp(coords, b)


def test_same_inputs_are_stable(coords):
    """Regression guard: uses the pre-existing signature, so it must pass either way."""
    x, y, z = coords
    assert _surface_coordinate_fingerprint(x, y, z) == _surface_coordinate_fingerprint(
        x.copy(), y.copy(), z.copy()
    )


def test_coordinates_still_matter(coords):
    """Regression guard: coordinate sensitivity must survive the change."""
    x, y, z = coords
    assert _surface_coordinate_fingerprint(x, y, z) != _surface_coordinate_fingerprint(
        x + 1.0, y, z
    )


def test_mask_is_optional():
    """Callers that pass no mask keep the previous behaviour."""
    rng = np.random.default_rng(1)
    x, y, z = (rng.random((H, W)).astype(np.float32) for _ in range(3))
    assert _surface_coordinate_fingerprint(x, y, z) == _surface_coordinate_fingerprint(x, y, z)
    assert _surface_coordinate_fingerprint(x, y, z) != _surface_coordinate_fingerprint(
        x, y, z, np.ones((H, W), dtype=bool)
    )
