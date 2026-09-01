"""threshold / scale / transpose must accept OME-Zarr input, including their own.

prepare() resolves the pyramid level with ZarrTask._get_input_array, which
handles groups, but the work items passed the *root* path to the workers, which
re-opened it with a bare zarr.open(). For an OME-Zarr that yields a Group, so
indexing raised KeyError and .ndim raised AttributeError, in every worker. Each
of these tasks writes an OME-Zarr group, so chaining any two of them was broken,
as was any OME-Zarr scroll volume as input.

The OME-Zarr cases here fail against the previous implementation; the plain
array cases pass both before and after and guard against a regression.
"""

import numpy as np
import pytest
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks.scale import ScaleConfig, ScaleTask
from vesuvius.image_proc.run.zarr_tasks.tasks.threshold import ThresholdConfig, ThresholdTask
from vesuvius.image_proc.run.zarr_tasks.tasks.transpose import TransposeConfig, TransposeTask

SHAPE = (8, 8, 8)


def _add_array(group, name, data, chunks):
    """Create one array inside a group, on zarr 2 or zarr 3.

    Group.create_dataset is zarr 2 only; zarr 3 replaced it with create_array.
    This package supports both (pyproject: zarr>=2.18.7,<4).
    """
    if hasattr(group, "create_dataset"):
        group.create_dataset(name, data=data, chunks=chunks)
        return
    array = group.create_array(
        name, shape=data.shape, chunks=chunks, dtype=data.dtype)
    array[:] = data


def _write_input(path, as_group):
    data = (np.random.default_rng(0).random(SHAPE) * 255).astype(np.uint8)
    if as_group:
        g = zarr.open_group(str(path), mode="w")
        _add_array(g, "0", data, (4, 4, 4))
        g.attrs["multiscales"] = [{"version": "0.4", "datasets": [{"path": "0"}]}]
    else:
        z = zarr.open(str(path), mode="w", shape=SHAPE, chunks=(4, 4, 4), dtype=np.uint8)
        z[:] = data
    return data


def _tasks():
    return {
        "threshold": lambda s, o: ThresholdTask(ThresholdConfig(
            input_zarr=s, output_zarr=o, num_workers=1, threshold=127.0, num_levels=1)),
        "scale": lambda s, o: ScaleTask(ScaleConfig(
            input_zarr=s, output_zarr=o, num_workers=1, scale_factor=2.0, num_levels=1)),
        "transpose": lambda s, o: TransposeTask(TransposeConfig(
            input_zarr=s, output_zarr=o, num_workers=1, transpose_order="xzy", num_levels=1)),
    }


@pytest.mark.parametrize("task_name", ["threshold", "scale", "transpose"])
@pytest.mark.parametrize("as_group", [True, False], ids=["ome_zarr_group", "plain_array"])
def test_task_accepts_both_input_layouts(tmp_path, task_name, as_group):
    src = tmp_path / "in.zarr"
    out = tmp_path / "out.zarr"
    _write_input(src, as_group)

    _tasks()[task_name](str(src), str(out)).run()

    written = zarr.open(str(out), mode="r")["0"]
    expected = (16, 16, 16) if task_name == "scale" else SHAPE
    assert tuple(written.shape) == expected


def test_output_can_be_fed_back_in(tmp_path):
    """Each task writes an OME-Zarr, so chaining two of them must work."""
    src = tmp_path / "in.zarr"
    first = tmp_path / "first.zarr"
    second = tmp_path / "second.zarr"
    _write_input(src, as_group=False)

    make = _tasks()["threshold"]
    make(str(src), str(first)).run()
    assert isinstance(zarr.open(str(first), mode="r"), zarr.Group)

    make(str(first), str(second)).run()

    out = np.asarray(zarr.open(str(second), mode="r")["0"][:])
    assert out.shape == SHAPE
    assert set(np.unique(out)) <= {0, 255}
