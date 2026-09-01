"""Representations of one physical segment must share one segment budget.

hierarchical_scroll_segment_weights says so in its own docstring:

    Multiple representations of the same physical segment intentionally share
    one segment budget. Their patch windows divide that budget instead of
    counting as independent segments.

It keyed on segment_relpath, which names one *rendering* of a segment, not the
segment. A segment rendered twice therefore got two budgets - twice the
sampling mass of a segment rendered once - which is the double-counting the
weighting exists to prevent. The audit reported the representation count as
"segments" too.

sampling_physical_segment_keys is the explicit mapping that answers this, and
FixedPriorBatchSampler in the same module already uses it.
"""

import pytest
import torch

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.training.samplers import (
    hierarchical_scroll_segment_weights,
)
from vesuvius.ink_detection.types import Patch, Segment


def build_config(tmp_path, physical_keys, scroll="A"):
    dataset = {
        "segments_path": str(tmp_path / "a"),
        "volume_scale": 0,
        "sampling_scroll": scroll,
        "sampling_representation_keys": {
            relpath: f"r{relpath}" for relpath in physical_keys
        },
    }
    if physical_keys is not None:
        dataset["sampling_physical_segment_keys"] = dict(physical_keys)
    return InkDataConfig.from_mapping({
        "mode": "flat",
        "patch_size": [1, 2, 2],
        "patch_overlap": 0.25,
        "patch_min_labeled_coverage": 0.0,
        "datasets": [dataset],
        "seed": 42,
        "sampling_strategy": "scroll_segment_balanced",
        "fixed_scroll_prior": {"seed": 42, "target_batch_counts": {scroll: 4}},
    })


def build_patches(config, tmp_path, counts):
    """counts: [(relpath, n_patches)]. Returns patches and each relpath's span."""
    patches, spans, start = [], {}, 0
    for relpath, count in counts:
        segment = Segment(
            data_config=config,
            source=config.datasets[0],
            dataset_idx=0,
            segment_relpath=relpath,
            segment_dir=tmp_path / relpath,
            segment_name=relpath,
            image_volume="unused",
        )
        patches.extend(
            Patch(segment=segment, bbox=(0, 0, 0, 1, 2, 2)) for _ in range(count)
        )
        spans[relpath] = (start, start + count)
        start += count
    return patches, spans


def mass_by_relpath(weights, spans):
    return {
        relpath: float(weights[lo:hi].sum()) for relpath, (lo, hi) in spans.items()
    }


def test_a_segment_rendered_twice_does_not_get_twice_the_mass(tmp_path):
    """A:1 is rendered as a1 and a2; A:2 once as a3. Both are one segment."""
    config = build_config(tmp_path, {"a1": "A:1", "a2": "A:1", "a3": "A:2"})
    patches, spans = build_patches(
        config, tmp_path, [("a1", 2), ("a2", 2), ("a3", 2)]
    )

    weights, audit = hierarchical_scroll_segment_weights(patches, config)
    mass = mass_by_relpath(weights, spans)

    assert mass["a1"] + mass["a2"] == pytest.approx(0.5)
    assert mass["a3"] == pytest.approx(0.5)
    assert audit["segments_per_scroll"] == {"A": 2}


def test_the_shared_budget_is_split_between_the_representations(tmp_path):
    """Uneven patch counts still divide one budget, not two."""
    config = build_config(tmp_path, {"a1": "A:1", "a2": "A:1", "a3": "A:2"})
    patches, spans = build_patches(
        config, tmp_path, [("a1", 1), ("a2", 5), ("a3", 2)]
    )

    weights, _ = hierarchical_scroll_segment_weights(patches, config)
    mass = mass_by_relpath(weights, spans)

    assert mass["a1"] + mass["a2"] == pytest.approx(0.5)
    assert mass["a3"] == pytest.approx(0.5)


def test_distinct_segments_still_get_equal_budgets(tmp_path):
    config = build_config(tmp_path, {"a1": "A:1", "a2": "A:2"})
    patches, spans = build_patches(config, tmp_path, [("a1", 1), ("a2", 7)])

    weights, audit = hierarchical_scroll_segment_weights(patches, config)
    mass = mass_by_relpath(weights, spans)

    assert mass["a1"] == pytest.approx(0.5)
    assert mass["a2"] == pytest.approx(0.5)
    assert audit["segments_per_scroll"] == {"A": 2}


def test_the_weights_still_sum_to_one(tmp_path):
    config = build_config(tmp_path, {"a1": "A:1", "a2": "A:1", "a3": "A:2"})
    patches, _ = build_patches(
        config, tmp_path, [("a1", 3), ("a2", 1), ("a3", 4)]
    )

    weights, _ = hierarchical_scroll_segment_weights(patches, config)

    assert float(weights.sum()) == pytest.approx(1.0)
    assert torch.all(weights > 0)


def test_a_relpath_with_no_mapping_is_refused(tmp_path):
    """Silently falling back would reintroduce the double count."""
    config = build_config(tmp_path, {"a1": "A:1"})
    patches, _ = build_patches(config, tmp_path, [("a1", 2), ("a2", 2)])

    with pytest.raises(ValueError, match="missing an explicit sampling mapping"):
        hierarchical_scroll_segment_weights(patches, config)


def test_configs_without_the_mapping_keep_working(tmp_path):
    """Nothing declared: each representation is its own segment, as before."""
    config = build_config(tmp_path, {})
    patches, spans = build_patches(config, tmp_path, [("a1", 2), ("a2", 6)])

    weights, audit = hierarchical_scroll_segment_weights(patches, config)
    mass = mass_by_relpath(weights, spans)

    assert mass["a1"] == pytest.approx(0.5)
    assert mass["a2"] == pytest.approx(0.5)
    assert audit["segments_per_scroll"] == {"A": 2}

