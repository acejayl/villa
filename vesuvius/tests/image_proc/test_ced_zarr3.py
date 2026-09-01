"""ced must run under zarr 3, which pyproject declares support for.

Two separate zarr-3 breakages met at the same two lines:

    compressor=zarr_in.compressor if hasattr(zarr_in, 'compressor') else None

1. **The guard does not guard.** Reading ``.compressor`` on a zarr-3 array
   raises ``TypeError``, and ``hasattr`` only swallows ``AttributeError`` - so
   the TypeError travels straight through the ``hasattr`` and out of ced.
2. **The output cannot be created.** zarr 3 defaults new arrays to the v3
   format, which rejects ``compressor`` outright:
   ``ValueError: compressor cannot be used for arrays with zarr_format 3``.

The second one fires even for a plain zarr-2 input, which is what this
package's own writers produce, so ``process_zarr_array`` cannot create its
output at all under zarr 3. It fails before a single slice is diffused.

These run on CPU. ced falls back to ``torch.device('cpu')`` when CUDA is
absent, so this needs no GPU.
"""

import numpy as np
import pytest
import zarr

zarr_v3 = int(zarr.__version__.split(".", 1)[0]) >= 3

CONFIG = {
    "lambda": 0.1,
    "sigma": 1.0,
    "rho": 1.0,
    "step_size": 0.1,
    "m": 1,
    "num_steps": 1,
}


@pytest.fixture
def no_diffusion(monkeypatch):
    """Run the real I/O path without the diffusion maths.

    The defect is entirely in how the output array is created; the solver is
    not under test and pulling it in would make these tests depend on numba,
    which does not install on Python 3.14. Patching the module attribute keeps
    process_zarr_array's own zarr code running exactly as shipped.
    """
    import torch

    from vesuvius.image_proc.run import ced

    # the real solver hands back a torch tensor, and ced calls .cpu() on it
    monkeypatch.setattr(
        ced, "run_coherence_diffusion",
        lambda image, *args, **kwargs: torch.as_tensor(
            np.asarray(image), dtype=torch.float32))


def _v2_input(path, shape=(2, 16, 16)):
    """A zarr-format-2 array, which is what this package's writers produce."""
    array = zarr.open(
        str(path), mode="w", shape=shape, chunks=(1, shape[1], shape[2]),
        dtype="uint8", **({"zarr_format": 2} if zarr_v3 else {}),
    )
    array[:] = np.random.default_rng(0).integers(
        0, 255, shape, dtype="uint8")
    return array


def test_a_v2_input_produces_an_output(tmp_path, no_diffusion):
    """The whole point: it currently cannot create the output array at all."""
    from vesuvius.image_proc.run.ced import process_zarr_array

    source, destination = tmp_path / "in.zarr", tmp_path / "out.zarr"
    _v2_input(source)

    process_zarr_array(str(source), str(destination), CONFIG, batch_size=1)

    written = zarr.open(str(destination), mode="r")
    assert written.shape == (2, 16, 16)


def test_the_output_stays_in_the_v2_format(tmp_path, no_diffusion):
    """Getting past the exception is not enough - it must not silently
    switch the store to v3, which nothing else in this package reads."""
    from vesuvius.image_proc.run.ced import process_zarr_array

    source, destination = tmp_path / "in.zarr", tmp_path / "out.zarr"
    _v2_input(source)

    process_zarr_array(str(source), str(destination), CONFIG, batch_size=1)

    written = zarr.open(str(destination), mode="r")
    assert getattr(written.metadata, "zarr_format", 2) == 2


def test_the_input_compressor_is_carried_over(tmp_path, no_diffusion):
    """Dropping compression would be a silent regression, not a fix."""
    from vesuvius.image_proc.run.ced import process_zarr_array

    source, destination = tmp_path / "in.zarr", tmp_path / "out.zarr"
    original = _v2_input(source)

    process_zarr_array(str(source), str(destination), CONFIG, batch_size=1)

    written = zarr.open(str(destination), mode="r")
    assert written.compressor is not None
    assert written.compressor == original.compressor


@pytest.mark.skipif(not zarr_v3, reason="the TypeError only exists on zarr 3")
def test_hasattr_does_not_guard_a_v3_compressor(tmp_path):
    """Why the original line could not work, pinned so nobody restores it.

    This passes on both trees deliberately - it characterises zarr, not our
    fix. The three tests above are the regression tests; this one exists so
    that the next person to write `hasattr(z, "compressor")` has a failing
    example to read.
    """
    v3 = zarr.open(str(tmp_path / "v3.zarr"), mode="w", shape=(2, 2),
                   dtype="uint8")

    # hasattr only swallows AttributeError, and this raises TypeError, so the
    # exception escapes the guard entirely.
    with pytest.raises(TypeError):
        hasattr(v3, "compressor") and v3.compressor
