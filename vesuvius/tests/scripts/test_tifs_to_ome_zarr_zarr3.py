"""tifs_to_ome_zarr has to run under zarr 3, which this package supports.

pyproject pins ``zarr>=2.18.7,<4``. Under zarr 3 the converter died before
writing anything:

    AttributeError: 'Group' object has no attribute 'create_dataset'

at both writer sites - the level-0 array and each downsampled pyramid level.
Group.create_dataset is a zarr 2 API; zarr 3 replaced it with create_array,
which additionally rejects ``compressor`` unless the array is explicitly
zarr_format 2. That is the format the .zattrs this script writes describes, so
asking for it is also what keeps the output readable by zarr 2.

The whole script was unusable on a supported dependency version.
"""

import json

import numpy as np
import tifffile
import zarr

SLICES = 4
SIZE = 64


def write_tiffs(directory):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(SLICES):
        plane = (rng.random((SIZE, SIZE)) * 255).astype(np.uint8)
        tifffile.imwrite(directory / f"{index:04d}.tif", plane)
    return directory


def convert(source, output, **kwargs):
    from vesuvius.scripts.tifs_to_ome_zarr import convert_tifs_to_ome_zarr

    return convert_tifs_to_ome_zarr(
        input_folder=source, output_path=output,
        chunk_shape=(2, 32, 32), **kwargs)


def test_conversion_runs(tmp_path):
    """The regression: this raised AttributeError under zarr 3."""
    source = write_tiffs(tmp_path / "in")
    output = tmp_path / "out.zarr"

    convert(source, output, num_levels=1, num_workers=1)

    level0 = zarr.open(str(output), mode="r")["0"]
    assert tuple(level0.shape) == (SLICES, SIZE, SIZE)


def test_the_pyramid_levels_are_written(tmp_path):
    """The second writer site, in the downsampling loop."""
    source = write_tiffs(tmp_path / "in")
    output = tmp_path / "out.zarr"

    convert(source, output, num_levels=2, num_workers=1)

    root = zarr.open(str(output), mode="r")
    assert tuple(root["0"].shape) == (SLICES, SIZE, SIZE)
    assert tuple(root["1"].shape) == (SLICES // 2, SIZE // 2, SIZE // 2)


def test_the_output_is_zarr_v2(tmp_path):
    """The .zattrs this script writes describes a v2 store, so it must be one."""
    source = write_tiffs(tmp_path / "in")
    output = tmp_path / "out.zarr"

    convert(source, output, num_levels=1, num_workers=1)

    assert (output / "0" / ".zarray").is_file(), "level 0 is not a zarr v2 array"
    assert (output / ".zattrs").is_file()
    metadata = json.loads((output / ".zattrs").read_text())
    assert "multiscales" in metadata


def test_the_pixels_survive(tmp_path):
    """Not just that it ran - that it wrote the right thing."""
    source = write_tiffs(tmp_path / "in")
    output = tmp_path / "out.zarr"
    expected = np.stack([
        np.asarray(tifffile.imread(source / f"{i:04d}.tif"))
        for i in range(SLICES)
    ])

    convert(source, output, num_levels=1, num_workers=1)

    written = np.asarray(zarr.open(str(output), mode="r")["0"][:])
    np.testing.assert_array_equal(written, expected)
