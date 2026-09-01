"""zero-range must never destroy its input when an output path is supplied.

ZeroRangeTask.run() rewrites config.input_zarr and ignores config.output_zarr.
The CLI exempted zero-range from the "output_zarr is required" check and only
refused --inplace *together with* an output path, so a plain two-positional
invocation passed validation, wrote nothing to the output, and zeroed the
input. The pre-flight overwrite prompt guarded the output path, which is never
touched, so it never fired for the path actually being modified.

This test fails against the previous implementation: the input is destroyed and
no exception is raised.
"""

import subprocess
import sys

import numpy as np
import pytest
import zarr


def _make_zarr(path, shape=(4, 4, 4)):
    data = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape) + 1
    z = zarr.open(str(path), mode="w", shape=shape, chunks=(2, 2, 2),
                  dtype=np.uint16, zarr_format=2)
    z[:] = data
    return data


def _run_cli(*argv):
    return subprocess.run(
        [sys.executable, "-m", "vesuvius.image_proc.run.zarr_tasks", *argv],
        capture_output=True, text=True, timeout=300,
    )


def test_output_path_is_rejected_instead_of_destroying_the_input(tmp_path):
    src = tmp_path / "precious_input.zarr"
    out = tmp_path / "wanted_output.zarr"
    original = _make_zarr(src)

    result = _run_cli(str(src), str(out), "--task", "zero-range",
                      "--z-start", "0", "--z-end", "3", "-n", "1")

    assert result.returncode != 0, (
        "CLI accepted an output path that zero-range ignores:\n" + result.stdout
    )
    assert "in place" in (result.stderr + result.stdout).lower()

    survived = np.asarray(zarr.open(str(src), mode="r")[:])
    np.testing.assert_array_equal(
        survived, original, err_msg="the input zarr was modified"
    )
    assert not out.exists(), "an output path was created despite the error"


def test_inplace_still_works(tmp_path):
    """The supported invocation must keep working."""
    src = tmp_path / "in.zarr"
    original = _make_zarr(src)

    result = _run_cli(str(src), "--task", "zero-range", "--inplace",
                      "--z-start", "0", "--z-end", "2", "-n", "1")

    if result.returncode != 0:
        pytest.skip(f"in-place run unavailable in this environment: {result.stderr[-400:]}")

    after = np.asarray(zarr.open(str(src), mode="r")[:])
    assert not np.array_equal(after, original), "in-place zero-range did nothing"
    # --z-end is inclusive: 0..2 are zeroed, 3 is untouched
    np.testing.assert_array_equal(after[:3], 0, err_msg="requested z-range not zeroed")
    np.testing.assert_array_equal(
        after[3:], original[3:], err_msg="zeroed outside the requested z-range"
    )
