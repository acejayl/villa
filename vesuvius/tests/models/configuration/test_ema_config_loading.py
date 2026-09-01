"""An `ema:` section in a training config has to actually reach the trainer.

_init_attributes read the EMA settings with
`getattr(self, "ema_config", {}) or {}`, but __init__ only ever loaded
tr_setup, tr_config, model_config and dataset_config from the YAML. Nothing
set self.ema_config, so the getattr fell back to {} on every run, ema_enabled
was False no matter what the config said, and the whole EMA path -
_create_ema_model, _update_ema_model, EMA validation, saving the EMA weights
into the checkpoint - was unreachable.

It failed silently: no error, no warning, training just quietly ran without
the averaging the config asked for.

The only places that set mgr.ema_config were tests, on stand-in managers that
never go through _init_attributes, which is why this was invisible from the
suite.
"""

import pytest
import yaml

from vesuvius.models.configuration.config_manager import ConfigManager

BASE = {
    "tr_setup": {"model_name": "test_model"},
    "tr_config": {"patch_size": [64, 64, 64]},
    "model_config": {},
    "dataset_config": {
        "targets": {"ink": {"out_channels": 1}},
    },
}


def write_config(tmp_path, extra=None):
    config = {key: dict(value) for key, value in BASE.items()}
    if extra:
        config.update(extra)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def load(tmp_path, extra=None):
    manager = ConfigManager(verbose=False)
    manager.load_config(str(write_config(tmp_path, extra)))
    return manager


def test_an_ema_section_turns_ema_on(tmp_path):
    manager = load(tmp_path, {"ema": {"enabled": True}})

    assert manager.ema_enabled is True


def test_ema_is_off_when_no_section_is_given(tmp_path):
    assert load(tmp_path).ema_enabled is False


def test_ema_settings_are_read_from_the_section(tmp_path):
    manager = load(tmp_path, {
        "ema": {
            "enabled": True,
            "decay": 0.995,
            "start_step": 500,
            "update_every_steps": 4,
        },
    })

    assert manager.ema_decay == pytest.approx(0.995)
    assert manager.ema_start_step == 500
    assert manager.ema_update_every_steps == 4


def test_validate_and_save_default_to_enabled(tmp_path):
    manager = load(tmp_path, {"ema": {"enabled": True}})

    assert manager.ema_validate is True
    assert manager.ema_save_in_checkpoint is True


def test_validate_can_be_turned_off_independently(tmp_path):
    manager = load(tmp_path, {"ema": {"enabled": True, "validate": False}})

    assert manager.ema_enabled is True
    assert manager.ema_validate is False


def test_the_section_round_trips_through_the_saved_config(tmp_path):
    """Inference reads it back under this key, so it has to be written there."""
    manager = load(tmp_path, {"ema": {"enabled": True, "validate": True}})

    saved = manager.convert_to_dict()

    assert saved["ema"]["enabled"] is True
    assert saved["ema"]["validate"] is True


def test_a_saved_config_reloads_with_the_same_ema_settings(tmp_path):
    manager = load(tmp_path, {"ema": {"enabled": True, "decay": 0.99}})

    round_tripped = tmp_path / "saved.yaml"
    round_tripped.write_text(
        yaml.safe_dump(manager.convert_to_dict()), encoding="utf-8"
    )
    reloaded = ConfigManager(verbose=False)
    reloaded.load_config(str(round_tripped))

    assert reloaded.ema_enabled is True
    assert reloaded.ema_decay == pytest.approx(0.99)


def test_inference_accepts_the_saved_shape(tmp_path):
    """The consumer's own check, run against what training now writes."""
    from vesuvius.models.run.inference import (
        _legacy_checkpoint_uses_ema_for_inference,
    )

    manager = load(tmp_path, {"ema": {"enabled": True, "validate": True}})
    checkpoint = {
        "config": manager.convert_to_dict(),
        "ema_model": {"some.weight": 0},
    }

    assert _legacy_checkpoint_uses_ema_for_inference(checkpoint) is True

