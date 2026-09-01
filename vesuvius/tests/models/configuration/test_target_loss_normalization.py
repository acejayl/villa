"""Every target must reach the trainer with a loss function.

_build_loss only fills a target's loss list inside `if "losses" in task_info`.
load_config never put a "losses" key on a target, so a target configured any
other way arrived without one and got an empty list - and an empty list is not
an error downstream, it is a constant zero:

    task_total_loss = torch.zeros(())
    for loss_fn, loss_weight in task_loss_fns:   # never runs
        ...

Zero gradient into that head, and a reported per-target loss of 0.0 that reads
as perfect convergence.

That covers the documented single-loss spelling. config_manager's own docstring
gives {"ink": {"out_channels": 1, "loss_fn": "BCEWithLogitsLoss", ...}} as the
example target, and _apply_loss_overrides converts loss_fn to a losses list when
restoring after a checkpoint load - but nothing did it on the way in.

set_targets_and_data was written to supply the default and has no callers.
"""

import pytest
import yaml

from vesuvius.models.configuration.config_manager import ConfigManager

DEFAULT_LOSS = "nnUNet_DC_and_CE_loss"


def load(tmp_path, target):
    config = {
        "tr_setup": {"model_name": "test_model"},
        "tr_config": {"patch_size": [64, 64, 64]},
        "model_config": {},
        "dataset_config": {"targets": {"ink": target}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manager = ConfigManager(verbose=False)
    manager.load_config(str(path))
    return manager


def loss_names(manager, target="ink"):
    return [entry["name"] for entry in manager.targets[target]["losses"]]


def test_the_documented_loss_fn_spelling_is_honoured(tmp_path):
    manager = load(tmp_path, {
        "out_channels": 1,
        "loss_fn": "BCEWithLogitsLoss",
        "activation": "sigmoid",
    })

    assert loss_names(manager) == ["BCEWithLogitsLoss"]


def test_a_target_with_no_loss_gets_the_default(tmp_path):
    manager = load(tmp_path, {"out_channels": 2})

    assert loss_names(manager) == [DEFAULT_LOSS]


def test_an_explicit_losses_list_is_left_alone(tmp_path):
    losses = [
        {"name": "SoftDiceLoss", "weight": 1.0},
        {"name": "BCEWithLogitsLoss", "weight": 0.5},
    ]
    manager = load(tmp_path, {"out_channels": 2, "losses": losses})

    assert manager.targets["ink"]["losses"] == losses


def test_every_target_ends_up_with_at_least_one_loss(tmp_path):
    """The invariant _build_loss depends on and never checked."""
    for target in (
        {"out_channels": 1, "loss_fn": "BCEWithLogitsLoss"},
        {"out_channels": 2},
        {"out_channels": 2, "losses": [{"name": "SoftDiceLoss", "weight": 1.0}]},
    ):
        manager = load(tmp_path, target)
        assert manager.targets["ink"]["losses"], f"no loss for {target}"


def test_a_target_with_an_empty_losses_list_is_refused(tmp_path):
    """Explicitly asking for no loss is a configuration error, not a no-op."""
    from vesuvius.models.training.train import BaseTrainer

    manager = load(tmp_path, {"out_channels": 2, "losses": []})
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.mgr = manager

    with pytest.raises(ValueError, match="no loss function"):
        trainer._build_loss()


@pytest.mark.parametrize(
    "name", ["is_unlabeled", "surface_skel", "ink_mask", "mask_ink", "plane_mask"]
)
def test_a_target_excluded_from_the_loss_may_have_none(tmp_path, name):
    """_should_include_target_in_loss skips these at training time, so demanding
    a loss for them here would reject configurations the trainer handles fine."""
    from vesuvius.models.training.train import BaseTrainer

    manager = load(tmp_path, {"out_channels": 2, "losses": []})
    manager.targets = {name: dict(manager.targets["ink"])}
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.mgr = manager

    assert trainer._build_loss() == {name: []}
