"""Validation metrics must cover every class the head predicts.

_initialize_evaluation_metrics read `task_config.get('num_classes', 2)`, but
nothing in the repo writes num_classes - a grep over models/configuration and
every shipped yaml finds zero hits. Configs declare `out_channels`. So every
metric was built with num_classes=2 no matter how wide the head was, while the
loss supervised all channels: the loss and the metric disagreed about the class
set.

Four shipped 3-class configs are affected - ps256_hzvt_msr.yaml
("out_channels: 3  # bg=0, hz=1, vt=2"), ps256_rectoverso_msr.yaml, and the two
pretrained_dino_pixelshuffle variants. On those runs iou_class_2/dice_class_2
were never emitted at all, and the logged mean_iou/mean_dice were means over
{0, 1} only - so the class the config exists to separate was invisible and the
headline number was inflated.

test_metrics_follow_out_channels calls the real method and fails against the
previous implementation.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vesuvius
from vesuvius.models.evaluation.iou_dice import IOUDiceMetric

CONFIG_ROOT = Path(vesuvius.__file__).parent / "models" / "configuration"


def _build_metrics(targets):
    """Call the real _initialize_evaluation_metrics without a full trainer.

    It reads self.mgr.targets and returns the metric dict; nothing else.
    """
    pytest.importorskip("pytorch_optimizer", reason="train.py imports it at module scope")
    from vesuvius.models.training.train import BaseTrainer

    stub = object.__new__(BaseTrainer)
    stub.mgr = SimpleNamespace(targets=targets)
    return BaseTrainer._initialize_evaluation_metrics(stub)


def _widths(metrics_for_task):
    return {
        type(m).__name__: getattr(m, "num_classes", None)
        for m in metrics_for_task
        if getattr(m, "num_classes", None) is not None
    }


@pytest.mark.parametrize("width", [3, 9])
def test_metrics_follow_out_channels(width):
    metrics = _build_metrics({"seg": {"out_channels": width}})

    widths = _widths(metrics["seg"])
    assert widths, "no metric exposed num_classes"
    for name, value in widths.items():
        assert value == width, f"{name} built with {value}, head is {width} wide"


def test_explicit_num_classes_still_wins():
    metrics = _build_metrics({"seg": {"num_classes": 4, "out_channels": 3}})
    assert all(v == 4 for v in _widths(metrics["seg"]).values())


def test_default_is_binary_when_neither_is_given():
    metrics = _build_metrics({"seg": {}})
    assert all(v == 2 for v in _widths(metrics["seg"]).values())


def test_no_shipped_config_declares_num_classes():
    """The premise: the key that was being read is never written."""
    offenders = [
        str(p.relative_to(CONFIG_ROOT))
        for p in CONFIG_ROOT.rglob("*.yaml")
        if "num_classes" in p.read_text(encoding="utf8")
    ]
    assert not offenders, f"unexpected num_classes in {offenders}"


def test_a_third_class_is_actually_scored():
    """Why the width matters: at 2 the class vanishes and the mean inflates."""
    gt = torch.tensor(np.array([[[[0, 1, 2, 2]]]]), dtype=torch.float32)
    pred = torch.tensor(np.array([[[[0, 1, 1, 0]]]]), dtype=torch.float32)

    two = IOUDiceMetric(num_classes=2).compute(pred=pred, gt=gt)
    three = IOUDiceMetric(num_classes=3).compute(pred=pred, gt=gt)

    assert "iou_class_2" not in two
    assert "iou_class_2" in three
    assert three["mean_iou"] < two["mean_iou"], (
        "dropping a badly-predicted class inflates the headline mean"
    )
