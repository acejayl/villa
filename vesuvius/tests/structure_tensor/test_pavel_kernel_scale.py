"""The 2D structure tensor must agree with the 3D one on identical data.

The 2D Pavel kernels were built by slicing the central plane out of the 3D
kernels. That plane still carries the leftover smoothing weight s[2] = 6 while
being divided by the second 16, so the 2D kernels came out 6/16 = 0.375x too
small and every tensor component 0.140625x too small. Directions survived;
magnitudes did not, and `compute_c2` in coherence-enhancing diffusion compares
the tensor magnitude against the user's absolute --lambda.

These tests fail against the previous implementation.
"""

import numpy as np
import pytest
import torch

from vesuvius.image_proc.geometry.structure_tensor import (
    StructureTensorComputer,
    _get_pavel_kernels_2d,
)

N = 32
INTERIOR = slice(8, 24)  # away from the padded border


def test_2d_matches_the_exact_analytic_value():
    """For f = 3x the structure tensor component Jxx is exactly 9."""
    x = np.arange(N)
    img = np.broadcast_to(3.0 * x[None, :], (N, N)).astype(np.float32)

    jxx = StructureTensorComputer(sigma=0.0).compute(
        torch.tensor(img), spatial_dims=2
    ).numpy()[2]

    assert jxx[INTERIOR, INTERIOR].mean() == pytest.approx(9.0, rel=1e-5)


def test_2d_and_3d_agree_on_the_same_gradient():
    x = np.arange(N)
    vol = np.broadcast_to(3.0 * x[None, None, :], (N, N, N)).astype(np.float32)
    img = np.broadcast_to(3.0 * x[None, :], (N, N)).astype(np.float32)

    computer = StructureTensorComputer(sigma=0.0)
    jxx_3d = computer.compute(torch.tensor(vol), spatial_dims=3).numpy()[5]
    jxx_2d = computer.compute(torch.tensor(img), spatial_dims=2).numpy()[2]

    ratio = jxx_2d[INTERIOR, INTERIOR].mean() / jxx_3d[INTERIOR, INTERIOR, INTERIOR].mean()
    assert ratio == pytest.approx(1.0, rel=1e-5), (
        f"2D and 3D disagree by {1 / ratio:.3f}x on identical data"
    )


def test_derivative_kernel_recovers_a_unit_slope():
    """A correctly normalized derivative kernel applied to a unit ramp gives 1."""
    _, kx = _get_pavel_kernels_2d(torch.device("cpu"), torch.float32)
    ramp = torch.arange(9, dtype=torch.float32).view(1, 9)
    assert float((kx[0, 0] * ramp).sum()) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("slope", [1.0, 2.5, -4.0])
def test_2d_gradient_magnitude_is_the_true_slope(slope):
    x = np.arange(N)
    img = np.broadcast_to(slope * x[None, :], (N, N)).astype(np.float32)

    jxx = StructureTensorComputer(sigma=0.0).compute(
        torch.tensor(img), spatial_dims=2
    ).numpy()[2]

    assert np.sqrt(jxx[INTERIOR, INTERIOR].mean()) == pytest.approx(abs(slope), rel=1e-5)
