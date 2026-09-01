"""Regression tests for the overwrite path of ``TifxyzWriter.write``.

Both tests fail against the previous implementation, which removed the
destination directory before moving the replacement into place.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile

from vesuvius.tifxyz import write_tifxyz
from vesuvius.tifxyz.types import Tifxyz
from vesuvius.tifxyz.writer import TifxyzWriter

SIZE = 16


def _surface(offset: float = 0.0) -> Tifxyz:
    grid = np.arange(SIZE * SIZE, dtype=np.float32).reshape(SIZE, SIZE) + offset
    return Tifxyz(
        _x=grid,
        _y=grid + 1000.0,
        _z=grid + 2000.0,
        uuid="test",
        _scale=(1.0, 1.0),
    )


def test_overwrite_preserves_labels_and_extra_channels(tmp_path):
    """An overwrite rewrites only the managed files and keeps everything else.

    Label images and extra channels are first-class objects in this package:
    ``write_extra_channel`` creates them and the reader enumerates them as
    ``Tifxyz._labels``. The round trip read -> edit -> write(overwrite=True)
    must not discard them.
    """
    seg = tmp_path / "seg"
    write_tifxyz(seg, _surface(), overwrite=False)

    TifxyzWriter(seg, overwrite=True).write_extra_channel(
        "normals_z", np.zeros((SIZE, SIZE), dtype=np.float32)
    )
    label = np.full((SIZE, SIZE), 7, dtype=np.uint8)
    tifffile.imwrite(seg / "label_ink.tif", label)

    write_tifxyz(seg, _surface(1.0), overwrite=True)

    assert (seg / "label_ink.tif").exists(), "overwrite deleted a label image"
    assert (seg / "normals_z.tif").exists(), "overwrite deleted an extra channel"
    np.testing.assert_array_equal(tifffile.imread(seg / "label_ink.tif"), label)

    # the managed files were still replaced
    np.testing.assert_allclose(tifffile.imread(seg / "x.tif"), _surface(1.0)._x)


def test_failed_overwrite_leaves_the_existing_surface_intact(tmp_path, monkeypatch):
    """A failure part-way through must not destroy the existing surface.

    Removing the destination first leaves a window in which neither the
    original nor the replacement is in place; a failure inside that window
    loses both.
    """
    seg = tmp_path / "seg"
    write_tifxyz(seg, _surface(), overwrite=False)

    original_x = np.array(tifffile.imread(seg / "x.tif"))
    before = sorted(p.name for p in seg.iterdir())

    real_replace = os.replace
    real_move = shutil.move

    def _is_new_surface(src) -> bool:
        return Path(src).name.startswith(".tifxyz_tmp_")

    def _replace(src, dst, *args, **kwargs):
        if _is_new_surface(src):
            raise OSError("simulated failure moving the replacement into place")
        return real_replace(src, dst, *args, **kwargs)

    def _move(src, dst, *args, **kwargs):
        if _is_new_surface(src):
            raise OSError("simulated failure moving the replacement into place")
        return real_move(src, dst, *args, **kwargs)

    # Only the step that puts the freshly written directory in place fails;
    # renaming the original aside and rolling it back still work.
    monkeypatch.setattr(os, "replace", _replace)
    monkeypatch.setattr(shutil, "move", _move)

    with pytest.raises(Exception):
        write_tifxyz(seg, _surface(1.0), overwrite=True)

    assert seg.is_dir(), "the existing surface directory was destroyed"
    assert sorted(p.name for p in seg.iterdir()) == before
    np.testing.assert_array_equal(tifffile.imread(seg / "x.tif"), original_x)

    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".tifxyz_tmp_")]
    assert leftover, "the newly written surface was discarded as well"
    assert (leftover[0] / "x.tif").exists()
