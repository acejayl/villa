"""OMEU8VectorWriter has to construct under zarr 3.

pyproject declares ``zarr>=2.18.7,<4``. Under zarr 3 the writer raises in
__init__, before anything can be written:

    AttributeError: 'Group' object has no attribute 'create_dataset'

Group.create_dataset is a zarr 2 API. zarr 3 replaced it with create_array,
which also rejects the numcodecs ``compressor`` this writer passes unless the
group is explicitly zarr_format 2 - and v2 is what the OME metadata the writer
emits describes, so asking for it is also what keeps the store correct.

The existing tests for this module only cover the encoding helpers, so nothing
constructed the writer.
"""

import numpy as np
import zarr

from vesuvius.structure_tensor.vf_format import OMEU8VectorWriter

SHAPE = (16, 16, 16)
CHUNKS = (8, 8, 8)


def test_the_writer_constructs(tmp_path):
    """The regression: this raised AttributeError under zarr 3."""
    writer = OMEU8VectorWriter(
        str(tmp_path / "vf.zarr"), "first_component", SHAPE, chunks_zyx=CHUNKS)

    assert tuple(writer.ds_z.shape) == SHAPE
    assert tuple(writer.ds_y.shape) == SHAPE
    assert tuple(writer.ds_x.shape) == SHAPE


def test_the_arrays_are_zarr_v2(tmp_path):
    """The OME metadata this writer emits describes a v2 store."""
    output = tmp_path / "vf.zarr"
    OMEU8VectorWriter(
        str(output), "first_component", SHAPE, chunks_zyx=CHUNKS)

    for axis in "zyx":
        assert (output / "first_component" / axis / "0" / ".zarray").is_file(), (
            f"{axis} scale is not a zarr v2 array"
        )


def test_downsampling_sizes_the_arrays(tmp_path):
    writer = OMEU8VectorWriter(
        str(tmp_path / "vf.zarr"), "first_component", SHAPE,
        chunks_zyx=CHUNKS, downsample=2)

    assert tuple(writer.ds_z.shape) == (8, 8, 8)


def test_the_confidence_scale_is_created_when_asked(tmp_path):
    writer = OMEU8VectorWriter(
        str(tmp_path / "vf.zarr"), "first_component", SHAPE,
        chunks_zyx=CHUNKS, make_confidence=True)

    assert writer.ds_conf is not None
    assert tuple(writer.ds_conf.shape) == SHAPE


def test_reopening_reuses_the_existing_scale(tmp_path):
    """_require_scale returns the existing dataset rather than recreating it."""
    output = tmp_path / "vf.zarr"
    first = OMEU8VectorWriter(
        str(output), "first_component", SHAPE, chunks_zyx=CHUNKS)
    first.ds_z[0, 0, 0] = 200

    second = OMEU8VectorWriter(
        str(output), "first_component", SHAPE, chunks_zyx=CHUNKS)

    assert int(np.asarray(second.ds_z[0, 0, 0])) == 200


def test_a_shape_mismatch_on_reopen_is_refused(tmp_path):
    output = tmp_path / "vf.zarr"
    OMEU8VectorWriter(str(output), "first_component", SHAPE, chunks_zyx=CHUNKS)

    try:
        OMEU8VectorWriter(
            str(output), "first_component", (8, 8, 8), chunks_zyx=CHUNKS)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("a mismatched shape was accepted")


def test_the_store_is_readable_afterwards(tmp_path):
    output = tmp_path / "vf.zarr"
    writer = OMEU8VectorWriter(
        str(output), "first_component", SHAPE, chunks_zyx=CHUNKS)
    writer.ds_x[:] = 7

    reopened = zarr.open_group(str(output), mode="r")

    assert int(np.asarray(reopened["first_component"]["x"]["0"][0, 0, 0])) == 7
