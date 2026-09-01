"""tifs_to_ome_zarr must not modify the TIFFs it is asked to read.

convert_tifs_to_ome_zarr called convert_tiffs_to_tiled unconditionally before
creating any output, and convert_to_tiled_tiff wrote a fresh single-page zstd
TIFF and then os.replace'd it over the user's source file. Striped (non-tiled)
2D slices are the ordinary scroll-export case, so this fired on nearly every
real run: every rewritten file lost its ImageDescription, resolution and other
tags, and the original bytes were unrecoverable.

Nothing said it would. The module docstring is "Convert a folder of 2D TIFFs to
a 6-level OME-Zarr pyramid" and --help describes input_folder only as "Folder
containing 2D TIFF files (one per z-slice)". There was no opt-out.

test_inputs_are_not_modified fails against the previous implementation: all
four inputs come back with new checksums and their descriptions replaced.
"""

import hashlib
import logging
import os
import tempfile

import numpy as np
import pytest
import zarr

tifffile = pytest.importorskip("tifffile")

from vesuvius.scripts.tifs_to_ome_zarr import convert_tifs_to_ome_zarr

SIZE = 300
NUM_SLICES = 4
SCRATCH_PREFIX = "tifs_to_ome_zarr_tiled_"


def _fingerprint(path):
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        description = getattr(page, "description", "")
        tiled = bool(getattr(page, "is_tiled", False))
    return {
        "sha": hashlib.sha1(open(path, "rb").read()).hexdigest(),
        "size": os.path.getsize(path),
        "description": description,
        "tiled": tiled,
    }


@pytest.fixture
def striped_input(tmp_path):
    """Ordinary non-tiled 2D slices, each carrying metadata worth keeping."""
    folder = tmp_path / "in"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for z in range(NUM_SLICES):
        tifffile.imwrite(
            folder / f"{z:05d}.tif",
            (rng.random((SIZE, SIZE)) * 255).astype(np.uint8),
            description=f"scroll1 slice z={z}",
            resolution=(1234, 1234),
        )
    return folder


def test_inputs_are_not_modified(striped_input, tmp_path, caplog):
    before = {p.name: _fingerprint(p) for p in sorted(striped_input.glob("*.tif"))}
    assert all(not f["tiled"] for f in before.values()), "fixture should be striped"

    with caplog.at_level(logging.INFO):
        convert_tifs_to_ome_zarr(striped_input, tmp_path / "out.zarr",
                                 num_levels=1, num_workers=1)

    after = {p.name: _fingerprint(p) for p in sorted(striped_input.glob("*.tif"))}
    modified = [name for name in before if before[name] != after[name]]
    assert not modified, f"the tool rewrote its own inputs: {modified}"

    # and specifically, the metadata survived
    for name, fp in after.items():
        assert fp["description"] == before[name]["description"]


def test_output_is_still_correct(striped_input, tmp_path):
    convert_tifs_to_ome_zarr(striped_input, tmp_path / "out.zarr",
                             num_levels=1, num_workers=1)

    level0 = zarr.open(str(tmp_path / "out.zarr"), mode="r")["0"]
    assert tuple(level0.shape) == (NUM_SLICES, SIZE, SIZE)

    expected = np.stack([
        tifffile.imread(striped_input / f"{z:05d}.tif") for z in range(NUM_SLICES)
    ])
    np.testing.assert_array_equal(np.asarray(level0[:]), expected)


def test_scratch_directory_is_cleaned_up(striped_input, tmp_path):
    before = {d for d in os.listdir(tempfile.gettempdir()) if d.startswith(SCRATCH_PREFIX)}

    convert_tifs_to_ome_zarr(striped_input, tmp_path / "out.zarr",
                             num_levels=1, num_workers=1)

    after = {d for d in os.listdir(tempfile.gettempdir()) if d.startswith(SCRATCH_PREFIX)}
    assert not (after - before), "tiled working copies were left behind"
