"""merge_predictions must not leak its temp memmaps.

merge_prediction_files() creates one np.memmap temp file per input prediction
plus one for the output. The cleanup flushed them but never closed them, then
called unlink(). While a mapping is open Windows locks the file, so every
unlink raised PermissionError [WinError 32] - and both call sites wrapped it in
`except Exception: pass`, so each merged output silently leaked its full-size
uint8 raster into %TEMP% forever. merge_predictions is a recursive CLI that
walks every preds folder, so a run over a scroll leaks GBs.

POSIX permits unlinking a mapped file, which is why ubuntu-only CI never saw
this. The same package already closes the mapping before unlinking in
image_proc/run/stack_composite_tifs.py.

This test fails against the previous implementation on Windows.
"""

import glob
import os
import tempfile

import numpy as np
import pytest

from vesuvius.ink_detection.preprocessing import merge_predictions as mp

Image = pytest.importorskip("PIL.Image", reason="Pillow needed to write inputs")


def _temp_memmaps():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "merge_predictions*.memmap")))


def test_merge_leaves_no_temp_memmaps_behind(tmp_path):
    before = _temp_memmaps()

    data = (np.random.default_rng(0).random((64, 48)) * 255).astype(np.uint8)
    inputs = []
    for name in ("a.png", "b.png"):
        p = tmp_path / name
        Image.fromarray(data).save(p)
        inputs.append(p)

    out = tmp_path / "merged.png"
    mp.merge_prediction_files(inputs, out, merge_method="soft_mean")

    assert out.exists(), "merge produced no output"

    leaked = _temp_memmaps() - before
    assert not leaked, (
        f"{len(leaked)} temp memmap(s) leaked into %TEMP%: "
        + ", ".join(sorted(os.path.basename(p) for p in leaked))
    )


def test_close_memmap_releases_the_file(tmp_path):
    """The helper must make the backing file deletable."""
    path = tmp_path / "scratch.memmap"
    array = np.memmap(path, mode="w+", dtype=np.uint8, shape=(32, 32))
    array[:] = 7

    mp.close_memmap(array)

    path.unlink()  # raises PermissionError on Windows if still mapped
    assert not path.exists()


def test_close_memmap_ignores_non_memmaps():
    mp.close_memmap(None)
    mp.close_memmap(np.zeros((4, 4), dtype=np.uint8))
