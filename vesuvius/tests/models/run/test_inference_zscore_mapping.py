"""A train.py 'zscore' model must be normalized per instance at inference.

train.py's 'zscore' is a per-instance z-score, unconditionally.
ZScoreNormalization.run calls

    normalize_zscore(image, mask=mask, use_mask=use_mask, target_dtype=...)

and never passes intensityproperties, so the mean and std come from the patch
being normalized. Only CTNormalization consumes the global statistics - it
asserts they are present.

Training nonetheless computes and stores intensity properties for 'zscore'
(zarr_dataset initializes them when the scheme is 'zscore' or 'ct'), so a
zscore checkpoint almost always carries a mean and std. Inference switched to
'global_zscore' on their mere presence, feeding the model dataset-level
statistics it had never seen. Silent, and on the default scheme -
config_manager defaults normalization_scheme to 'zscore'.

inference.py already had the correct mapping for the nnU-Net path,
_NNUNET_NORMALIZATION_MAP['ZScoreNormalization'] == 'instance_zscore'; the
train.py branch contradicted it.
"""

from unittest import mock

import numpy as np
import pytest

from vesuvius.models.run.inference import (
    _NNUNET_NORMALIZATION_MAP,
    Inferer,
)
from vesuvius.models.training.normalization import get_normalization

PROPS = {
    "mean": 1000.0,
    "std": 500.0,
    "percentile_00_5": 0.0,
    "percentile_99_5": 2000.0,
}


def test_training_zscore_ignores_intensity_properties():
    """The premise: this is what the model was actually trained on."""
    patch = np.full((4, 4, 4), 50.0, dtype=np.float32)
    patch[0, 0, 0] = 60.0

    normalized = get_normalization("zscore", PROPS).run(patch)

    # Per-instance: the patch is standardized to its own statistics.
    assert float(normalized.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(normalized.std()) == pytest.approx(1.0, abs=1e-4)

    # A global z-score would have produced this instead, which is nowhere near.
    global_mean = (float(patch.mean()) - PROPS["mean"]) / PROPS["std"]
    assert abs(global_mean) > 1.0


def build(tmp_path, intensity_properties):
    inferer = Inferer(
        model_path=str(tmp_path / "model.pth"),
        input_dir=str(tmp_path / "in.zarr"),
        output_dir=str(tmp_path / "out"),
        patch_size=(64, 64, 64),
        verbose=False,
    )
    inferer.model_normalization_scheme = "zscore"
    inferer.model_intensity_properties = intensity_properties
    inferer.normalization_scheme = "zscore"
    return inferer


def dataset_kwargs(inferer):
    with mock.patch(
        "vesuvius.models.run.inference.VCDataset", autospec=True
    ) as dataset:
        try:
            inferer._create_dataset_and_loader()
        except Exception:
            pass  # everything past the construction needs a real volume
        assert dataset.call_args, "VCDataset was never constructed"
        return dataset.call_args.kwargs


def test_zscore_with_intensity_properties_stays_per_instance(tmp_path):
    """The regression: their presence must not switch the scheme."""
    kwargs = dataset_kwargs(build(tmp_path, PROPS))

    assert kwargs["normalization_scheme"] == "instance_zscore"


def test_zscore_without_intensity_properties_is_per_instance(tmp_path):
    kwargs = dataset_kwargs(build(tmp_path, None))

    assert kwargs["normalization_scheme"] == "instance_zscore"


def test_global_statistics_are_not_passed_for_a_zscore_model(tmp_path):
    """Training never saw them, so inference must not apply them."""
    kwargs = dataset_kwargs(build(tmp_path, PROPS))

    assert kwargs.get("global_mean") is None
    assert kwargs.get("global_std") is None


def test_the_two_normalization_paths_agree(tmp_path):
    """The nnU-Net path already mapped this correctly; now both do."""
    assert _NNUNET_NORMALIZATION_MAP["ZScoreNormalization"] == "instance_zscore"
    assert dataset_kwargs(build(tmp_path, PROPS))["normalization_scheme"] == (
        _NNUNET_NORMALIZATION_MAP["ZScoreNormalization"]
    )
