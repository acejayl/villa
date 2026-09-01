"""--overlap must mean overlap, not stride.

VCDataset takes step_size, documented in compute_steps_for_sliding_window as
"step size as a fraction (0 <= step_size_factor <= 1), 0 means no overlap (full
stride)". The inferer passed its overlap straight into that parameter, so the
two meanings were swapped: --overlap 0.75, asking for a dense run, produced a
25%-overlap sparse one instead, and --overlap 0.1 produced a 90%-overlap run
roughly an order of magnitude slower than intended.

They agree at 0.5, which is the default, and at 0, which the callee special
cases - so the two values anyone would reach for first were the two that hid it.

The check here is on the stride the dataset is actually built with.
"""

from unittest import mock

import pytest

from vesuvius.models.run.inference import Inferer


def build(tmp_path, overlap):
    return Inferer(
        model_path=str(tmp_path / "model.pth"),
        input_dir=str(tmp_path / "in.zarr"),
        output_dir=str(tmp_path / "out"),
        overlap=overlap,
        patch_size=(64, 64, 64),
        verbose=False,
    )


def stride_fraction_for(tmp_path, overlap):
    """Build the dataset and report the step_size VCDataset was handed."""
    inferer = build(tmp_path, overlap)
    with mock.patch(
        "vesuvius.models.run.inference.VCDataset", autospec=True
    ) as dataset:
        try:
            inferer._create_dataset_and_loader()
        except Exception:
            # Everything past the VCDataset construction needs a real volume.
            pass
        assert dataset.call_args, "VCDataset was never constructed"
        return dataset.call_args.kwargs["step_size"]


@pytest.mark.parametrize(
    "overlap,expected_stride",
    [
        (0.0, 1.0),    # tile edge to edge
        (0.25, 0.75),
        (0.5, 0.5),    # the default, where the two conventions coincide
        (0.75, 0.25),  # a dense run: quarter-patch steps
        (0.9, 0.1),
    ],
)
def test_overlap_becomes_the_complementary_stride(
    tmp_path, overlap, expected_stride
):
    assert stride_fraction_for(tmp_path, overlap) == pytest.approx(
        expected_stride
    )


def test_more_overlap_means_a_shorter_stride(tmp_path):
    """The direction alone: the old code had this backwards."""
    strides = [stride_fraction_for(tmp_path, o) for o in (0.1, 0.4, 0.6, 0.9)]
    assert strides == sorted(strides, reverse=True), (
        f"stride must shrink as overlap grows, got {strides}"
    )


def test_an_overlap_of_one_is_rejected(tmp_path):
    """It would mean a zero stride, so the window would never advance."""
    with pytest.raises(ValueError, match="overlap"):
        build(tmp_path, 1.0)


@pytest.mark.parametrize("overlap", [-0.1, 1.5])
def test_out_of_range_overlaps_are_rejected(tmp_path, overlap):
    with pytest.raises(ValueError, match="overlap"):
        build(tmp_path, overlap)


@pytest.mark.parametrize("overlap", [0.0, 0.5, 0.99])
def test_valid_overlaps_are_accepted(tmp_path, overlap):
    assert build(tmp_path, overlap).overlap == pytest.approx(overlap)
