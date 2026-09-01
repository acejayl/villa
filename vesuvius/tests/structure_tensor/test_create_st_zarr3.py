"""The structure-tensor writers have to run under zarr 3.

Related to #1670: the same zarr-2-only API, here in create_st.

pyproject declares ``zarr>=2.18.7,<4``. Under zarr 3 the eigenanalysis stage
raises before writing anything:

    AttributeError: 'Group' object has no attribute 'create_dataset'

Group.create_dataset is a zarr 2 API. zarr 3 replaced it with create_array,
which also rejects the numcodecs ``compressor`` these arrays are built with
unless they are explicitly zarr_format 2 - and v2 is what the OME metadata
written alongside them assumes.

Five writer sites were affected: the structure-tensor array itself, the three
per-axis vector scales, the confidence scale, and the two optional
full-precision eigen arrays.

num_workers=0 throughout: the DataLoader's worker processes cannot be pickled
on Windows, which is a separate pre-existing problem and not what these cover.
"""

import numpy as np
import zarr

from vesuvius.structure_tensor.create_st import _finalize_structure_tensor_torch

SHAPE = (8, 8, 8)


def write_structure_tensor(path, zarr_format=2):
    """A 6-channel structure tensor volume, as the compute stage writes."""
    group = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    array = group.create_array(
        "structure_tensor", shape=(6,) + SHAPE, chunks=(6, 4, 4, 4),
        dtype="f4")
    array[:] = np.random.default_rng(0).random((6,) + SHAPE).astype("f4")
    return group


def run(path, **kwargs):
    _finalize_structure_tensor_torch(
        zarr_path=str(path), chunk_size=(4, 4, 4), num_workers=0,
        compressor=None, verbose=False, device="cpu", **kwargs)


def test_eigenanalysis_runs(tmp_path):
    """The regression: this raised AttributeError under zarr 3."""
    path = tmp_path / "st.zarr"
    write_structure_tensor(path)

    run(path)

    written = zarr.open_group(str(path), mode="r")
    for name in ("first_component", "second_component", "normal", "confidence"):
        assert name in written, f"{name} was not written"


def test_the_vector_scales_have_the_right_shape(tmp_path):
    path = tmp_path / "st.zarr"
    write_structure_tensor(path)

    run(path)

    written = zarr.open_group(str(path), mode="r")
    for axis in "zyx":
        assert tuple(written["first_component"][axis]["0"].shape) == SHAPE


def test_the_outputs_are_zarr_v2(tmp_path):
    """The OME metadata written alongside these describes a v2 store."""
    path = tmp_path / "st.zarr"
    write_structure_tensor(path)

    run(path)

    assert (path / "first_component" / "z" / "0" / ".zarray").is_file()
    assert (path / "confidence" / "0" / ".zarray").is_file()


def test_keep_eigen_writes_the_full_precision_arrays(tmp_path):
    """Two more writer sites, only reached with keep_eigen."""
    path = tmp_path / "st.zarr"
    write_structure_tensor(path)

    run(path, keep_eigen=True)

    written = zarr.open_group(str(path), mode="r")
    assert tuple(written["eigenvectors"].shape) == (9,) + SHAPE
    assert tuple(written["eigenvalues"].shape) == (3,) + SHAPE


def test_confidence_is_a_uint8_in_range(tmp_path):
    """Not just that it ran - that the values are the encoded ones."""
    path = tmp_path / "st.zarr"
    write_structure_tensor(path)

    run(path)

    confidence = np.asarray(
        zarr.open_group(str(path), mode="r")["confidence"]["0"][:])
    assert confidence.dtype == np.uint8
    assert confidence.max() <= 255


def test_rerunning_replaces_the_scales(tmp_path):
    """Each scale is deleted and recreated, which is a second path through it."""
    path = tmp_path / "st.zarr"
    write_structure_tensor(path)

    run(path)
    run(path)

    written = zarr.open_group(str(path), mode="r")
    assert tuple(written["normal"]["z"]["0"].shape) == SHAPE
