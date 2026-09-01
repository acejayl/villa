"""zarr_tasks has to run under zarr 3, which this package declares support for.

pyproject pins ``zarr>=2.18.7,<4``. Under zarr 3 every task failed immediately,
in one of two places depending on the input:

  * a **zarr-format-2 input** reached the output writer. create_level_dataset
    builds it through ``zarr.NestedDirectoryStore``, and edt_dilate calls
    ``Group.create_dataset`` directly - both removed in zarr 3 ->
    ``AttributeError: module 'zarr' has no attribute 'NestedDirectoryStore'``
    and ``AttributeError: 'Group' object has no attribute 'create_dataset'``
  * a **zarr-format-3 input** failed earlier, in prepare(), reading
    ``input_z.compressor`` -> ``TypeError: `compressor` is not available for
    Zarr format 3 arrays``

and format 3 is what plain ``zarr.open(..., mode="w")`` produces under zarr 3,
so an array created with default settings is not readable by these tasks.

Note the second one is a **TypeError**, so the ``hasattr(z, "compressor")``
idiom does not guard it - hasattr only swallows AttributeError.

Both directions are covered here. On zarr 2 these all passed already and this
guards against a regression.
"""

import json

import numpy as np
import pytest
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks.merge import MergeConfig, MergeTask
from vesuvius.image_proc.run.zarr_tasks.tasks.recompress import (
    RecompressConfig,
    RecompressTask,
)
from vesuvius.image_proc.run.zarr_tasks.tasks.scale import ScaleConfig, ScaleTask
from vesuvius.image_proc.run.zarr_tasks.tasks.threshold import (
    ThresholdConfig,
    ThresholdTask,
)
from vesuvius.image_proc.run.zarr_tasks.tasks.edt_dilate import (
    EdtDilateConfig,
    EdtDilateTask,
)
from vesuvius.image_proc.run.zarr_tasks.tasks.transpose import (
    TransposeConfig,
    TransposeTask,
)


def input_compressor(array):
    """Imported in-test so the task cases still collect on a tree without it."""
    from vesuvius.image_proc.run.zarr_tasks.utils import (
        input_compressor as helper,
    )

    return helper(array)


_ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3
SHAPE = (8, 8, 8)


def write_input(path, zarr_format):
    data = (np.random.default_rng(0).random(SHAPE) * 255).astype(np.uint8)
    kwargs = {}
    if _ZARR_V3:
        kwargs["zarr_format"] = zarr_format
    elif zarr_format == 3:
        pytest.skip("zarr 2 cannot write the v3 format")
    array = zarr.open(
        str(path), mode="w", shape=SHAPE, chunks=(4, 4, 4),
        dtype=np.uint8, **kwargs,
    )
    array[:] = data
    return data


TASKS = {
    "threshold": (
        lambda s, o: ThresholdTask(ThresholdConfig(
            input_zarr=s, output_zarr=o, num_workers=1,
            threshold=127.0, num_levels=1)),
        SHAPE,
    ),
    "scale": (
        lambda s, o: ScaleTask(ScaleConfig(
            input_zarr=s, output_zarr=o, num_workers=1,
            scale_factor=2.0, num_levels=1)),
        (16, 16, 16),
    ),
    "transpose": (
        lambda s, o: TransposeTask(TransposeConfig(
            input_zarr=s, output_zarr=o, num_workers=1,
            transpose_order="xzy", num_levels=1)),
        SHAPE,
    ),
}


@pytest.mark.parametrize("task_name", sorted(TASKS))
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_task_runs_for_either_input_format(tmp_path, task_name, zarr_format):
    build, expected_shape = TASKS[task_name]
    source = tmp_path / f"{task_name}{zarr_format}_in.zarr"
    output = tmp_path / f"{task_name}{zarr_format}_out.zarr"
    write_input(source, zarr_format)

    build(source, output).run()

    written = np.asarray(zarr.open(str(output), mode="r")["0"][:])
    assert written.shape == expected_shape


def test_the_output_is_zarr_v2_with_nested_chunk_keys(tmp_path):
    """The .zgroup the writer hand-writes says v2, so the arrays must be v2."""
    source, output = tmp_path / "in.zarr", tmp_path / "out.zarr"
    write_input(source, 2)

    TASKS["threshold"][0](source, output).run()

    level = output / "0"
    assert (level / ".zarray").is_file(), "level 0 is not a zarr v2 array"
    chunk_keys = [
        p.relative_to(level).as_posix()
        for p in level.rglob("*") if p.is_file() and not p.name.startswith(".")
    ]
    assert chunk_keys, "no chunks were written"
    assert all("/" in key for key in chunk_keys), (
        f"chunk keys are not nested: {chunk_keys[:4]}"
    )


def test_reading_a_compressor_never_raises(tmp_path):
    """hasattr does not guard this: zarr 3 raises TypeError, not AttributeError."""
    for zarr_format in (2, 3):
        path = tmp_path / f"c{zarr_format}.zarr"
        write_input(path, zarr_format)
        array = zarr.open(str(path), mode="r")

        input_compressor(array)  # must not raise for either format


def test_a_v3_array_reports_no_compressor(tmp_path):
    if not _ZARR_V3:
        pytest.skip("needs zarr 3 to make a v3 array")
    path = tmp_path / "v3.zarr"
    write_input(path, 3)

    assert input_compressor(zarr.open(str(path), mode="r")) is None


def write_group_input(path, zarr_format):
    """edt_dilate takes a resolution level, so its input is a group."""
    if not _ZARR_V3 and zarr_format == 3:
        pytest.skip("zarr 2 cannot write the v3 format")
    kwargs = {"zarr_format": zarr_format} if _ZARR_V3 else {}
    group = zarr.open_group(str(path), mode="w", **kwargs)
    volume = np.zeros(SHAPE, dtype=np.uint8)
    volume[4, 4, 4] = 1
    if hasattr(group, "create_dataset"):
        group.create_dataset("0", data=volume, chunks=(4, 4, 4))
    else:
        array = group.create_array(
            "0", shape=volume.shape, chunks=(4, 4, 4), dtype=volume.dtype)
        array[:] = volume
    return volume


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_edt_dilate_runs_for_either_input_format(tmp_path, zarr_format):
    """It writes its output itself rather than through create_level_dataset,
    calling Group.create_dataset, which zarr 3 removed."""
    source = tmp_path / f"edt{zarr_format}_in.zarr"
    output = tmp_path / f"edt{zarr_format}_out.zarr"
    write_group_input(source, zarr_format)

    EdtDilateTask(EdtDilateConfig(
        input_zarr=source, output_zarr=output, num_workers=1,
        distance=2.0, chunk_size=(4, 4, 4), resolution="0")).run()

    written = np.asarray(zarr.open(str(output), mode="r")["0"][:])
    assert written.shape == SHAPE
    assert int((written > 0).sum()) > 1, "the seed voxel was not dilated"


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_merge_runs_for_either_input_format(tmp_path, zarr_format):
    """merge writes its own output too, at two sites."""
    first = tmp_path / f"m{zarr_format}_a.zarr"
    second = tmp_path / f"m{zarr_format}_b.zarr"
    output = tmp_path / f"m{zarr_format}_out.zarr"
    write_group_input(first, zarr_format)
    write_group_input(second, zarr_format)

    MergeTask(MergeConfig(
        input_zarr=first, output_zarr=output, num_workers=1,
        input2_zarr=second, num_levels=1, level="0")).run()

    written = np.asarray(zarr.open(str(output), mode="r")["0"][:])
    assert written.shape == SHAPE


def test_recompress_in_place_runs_and_keeps_the_data(tmp_path):
    """Issue #1670: _run_inplace builds its temp array with a bare
    zarr.open(..., compressor=...), which zarr 3 rejects outright."""
    source = tmp_path / "recompress.zarr"
    expected = write_group_input(source, 2)

    RecompressTask(RecompressConfig(
        input_zarr=source, output_zarr=None, num_workers=1, inplace=True)).run()

    written = np.asarray(zarr.open(str(source), mode="r")["0"][:])
    np.testing.assert_array_equal(written, expected)


def test_recompressing_a_v3_store_in_place_is_refused(tmp_path):
    """Rather than leaving a v2 array inside a v3 group, which cannot be read.

    A numcodecs compressor only applies to v2 arrays, so there is nothing
    correct to write here - saying so beats corrupting the store.
    """
    if not _ZARR_V3:
        pytest.skip("needs zarr 3 to make a v3 store")
    source = tmp_path / "v3store.zarr"
    write_group_input(source, 3)

    with pytest.raises(NotImplementedError, match="v3"):
        RecompressTask(RecompressConfig(
            input_zarr=source, output_zarr=None, num_workers=1,
            inplace=True)).run()


def write_v2_level(path, separator):
    """One v2 array on disk with an explicit chunk-key separator."""
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    level = path / "0"
    kwargs = {"zarr_format": 2} if _ZARR_V3 else {}
    array = zarr.open(
        str(level), mode="w", shape=SHAPE, chunks=(4, 4, 4), dtype="u1",
        dimension_separator=separator, **kwargs)
    array[:] = 5
    return level


def chunk_keys(level):
    return sorted(
        p.relative_to(level).as_posix()
        for p in level.rglob("*") if p.is_file() and not p.name.startswith(".")
    )


@pytest.mark.parametrize("separator", [".", "/"])
def test_recompress_keeps_the_stores_chunk_key_layout(tmp_path, separator):
    """A replacement array must not silently change the separator.

    The temp array recompression builds replaces the level in place, so if it
    is created with a different separator the store's chunk keys change shape
    underneath the caller. It reads correctly either way - .zarray records the
    separator - but the files on disk are not the ones that were there.
    """
    source = tmp_path / f"sep{'dot' if separator == '.' else 'slash'}.zarr"
    level = write_v2_level(source, separator)
    before = chunk_keys(level)

    RecompressTask(RecompressConfig(
        input_zarr=source, output_zarr=None, num_workers=1, inplace=True)).run()

    assert chunk_keys(level) == before, "the chunk-key separator changed"
    assert np.all(np.asarray(zarr.open(str(level), mode="r")[:]) == 5)
