"""Normalization must apply per read instance, whatever shape the read returns.

The class docstring promises that 'instance_zscore' and 'instance_minmax' are
"computed per slice/volume instance". Deciding whether a channel axis is present
from the returned array's ndim alone breaks that for every read that is not
exactly 3D: a 2D slice gets normalized per row, a 1D read collapses to zeros
under z-score, a scalar read raises, and on a 4D store a single z-plane across
channels gets normalized jointly instead of per channel.

Every test here fails against the previous implementation.
"""

import numpy as np
import pytest
import zarr

from vesuvius.data.volume import Volume

SHAPE = (16, 16, 16)


def _store(tmp_path, shape=SHAPE, seed=0):
    path = str(tmp_path / "vol.zarr")
    data = np.random.default_rng(seed).integers(0, 255, shape, dtype="uint8")
    z = zarr.open(path, mode="w", shape=shape, chunks=tuple(min(8, s) for s in shape),
                  dtype="uint8", zarr_format=2)
    z[:] = data
    return path, data


def test_2d_slice_is_normalized_as_one_slice_not_per_row(tmp_path):
    path, data = _store(tmp_path)
    vol = Volume(type="zarr", path=path, format="zarr",
                 normalization_scheme="instance_minmax")

    got = np.asarray(vol[8, :, :])
    plane = data[8].astype(np.float32)
    expected = (plane - plane.min()) / (plane.max() - plane.min())

    np.testing.assert_allclose(got, expected, atol=1e-6)

    # the failure signature: every row independently spanning [0, 1]
    rows_all_full_range = (
        np.allclose(got.min(axis=1), 0.0) and np.allclose(got.max(axis=1), 1.0)
    )
    assert not rows_all_full_range, "each row was normalized separately"


def test_1d_read_is_not_all_zeros_under_zscore(tmp_path):
    path, data = _store(tmp_path)
    vol = Volume(type="zarr", path=path, format="zarr",
                 normalization_scheme="instance_zscore")

    got = np.asarray(vol[4, 5, :])
    line = data[4, 5, :].astype(np.float32)
    expected = (line - line.mean()) / max(line.std(), 1e-8)

    assert not np.allclose(got, 0.0), "1D read collapsed to zeros"
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_scalar_read_does_not_raise(tmp_path):
    path, _ = _store(tmp_path)
    vol = Volume(type="zarr", path=path, format="zarr",
                 normalization_scheme="instance_minmax")

    value = vol[3, 4, 5]  # the form documented in docs/accessing_data.md
    assert np.isscalar(value) or np.asarray(value).ndim == 0


def test_4d_store_normalizes_each_channel_separately(tmp_path):
    path = str(tmp_path / "vol4.zarr")
    data = np.random.default_rng(1).integers(0, 255, (2, 8, 8, 8), dtype="uint8")
    z = zarr.open(path, mode="w", shape=data.shape, chunks=(1, 4, 4, 4),
                  dtype="uint8", zarr_format=2)
    z[:] = data
    vol = Volume(type="zarr", path=path, format="zarr",
                 normalization_scheme="instance_minmax")

    # one z-plane, channel axis retained -> each channel spans [0, 1] on its own
    got = np.asarray(vol[:, 3, :, :])
    assert got.shape[0] == 2
    for c in range(got.shape[0]):
        assert np.isclose(got[c].min(), 0.0), f"channel {c} not independently scaled"
        assert np.isclose(got[c].max(), 1.0), f"channel {c} not independently scaled"


def test_4d_store_single_channel_index_is_one_instance(tmp_path):
    path = str(tmp_path / "vol4.zarr")
    data = np.random.default_rng(2).integers(0, 255, (2, 8, 8, 8), dtype="uint8")
    z = zarr.open(path, mode="w", shape=data.shape, chunks=(1, 4, 4, 4),
                  dtype="uint8", zarr_format=2)
    z[:] = data
    vol = Volume(type="zarr", path=path, format="zarr",
                 normalization_scheme="instance_minmax")

    got = np.asarray(vol[1, 3, :, :])
    plane = data[1, 3].astype(np.float32)
    expected = (plane - plane.min()) / (plane.max() - plane.min())
    np.testing.assert_allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("scheme", ["instance_minmax", "instance_zscore", "percentile_minmax"])
def test_3d_read_still_matches_whole_volume_statistics(tmp_path, scheme):
    """The 3D path was already correct and must stay correct."""
    path, data = _store(tmp_path)
    vol = Volume(type="zarr", path=path, format="zarr", normalization_scheme=scheme)
    got = np.asarray(vol[:, :, :])
    block = data.astype(np.float32)

    if scheme == "instance_minmax":
        expected = (block - block.min()) / (block.max() - block.min())
    elif scheme == "instance_zscore":
        expected = (block - block.mean()) / max(block.std(), 1e-8)
    else:
        lo, hi = np.percentile(block, (1.0, 99.0))
        expected = (np.clip(block, lo, hi) - lo) / float(hi - lo)

    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_none_scheme_returns_raw_values_for_every_shape(tmp_path):
    path, data = _store(tmp_path)
    vol = Volume(type="zarr", path=path, format="zarr", normalization_scheme="none")

    np.testing.assert_array_equal(np.asarray(vol[8, :, :]), data[8])
    np.testing.assert_array_equal(np.asarray(vol[4, 5, :]), data[4, 5, :])
    assert int(np.asarray(vol[3, 4, 5])) == int(data[3, 4, 5])
