"""Hole counts are only meaningful for complementary connectivity pairs.

compute_component_topology counts handles as b1 = b0 + b2 - chi, mixing a
foreground Euler characteristic with a background cavity count. That identity
only holds when the two connectivities are complementary - a Jordan pair. The
CLI defaulted both to 6 and offered every combination of 6/18/26, so the
default run reported topologically meaningless numbers.

The witness is a 5x5x5 shell with one corner voxel of its wall removed, so the
interior reaches the outside through a single corner contact and the shape has
neither a handle nor a cavity:

    (6, 26) Jordan pair     -> b1 = 0, b2 = 0   correct
    (6, 6)  the old default -> b1 = 1, b2 = 1   a handle and a cavity that do not exist
    (26, 26)                -> b1 = -1          not a Betti number at all

Separately, --fg-connectivity offered 18, which no Euler-number backend
implements for 3D; choosing it raised NotImplementedError partway through a run.
"""

import numpy as np
import pytest

pytest.importorskip("skimage")

from vesuvius.scripts.detect_holes import (  # noqa: E402
    compute_component_topology,
)

# The contract under test, spelled out here rather than imported so the
# behavioural assertions below still run against a tree that lacks the constant.
JORDAN_PAIRS = {(6, 26), (26, 6)}


def check_connectivity_pair(fg, bg):
    from vesuvius.scripts.detect_holes import check_connectivity_pair as check

    return check(fg, bg)


def leaky_shell():
    """A hollow cube whose interior escapes through one corner contact."""
    volume = np.zeros((9, 9, 9), dtype=bool)
    volume[2:7, 2:7, 2:7] = True
    volume[3:6, 3:6, 3:6] = False
    volume[2, 2, 2] = False
    return volume


def sealed_shell():
    """The same cube, intact: one cavity, no handle."""
    volume = np.zeros((9, 9, 9), dtype=bool)
    volume[2:7, 2:7, 2:7] = True
    volume[3:6, 3:6, 3:6] = False
    return volume


def solid_torus(size=24, major=6, minor=2):
    """One handle, no cavity."""
    grid = np.mgrid[0:size, 0:size, 0:size] - size // 2
    z, y, x = grid
    distance = (np.sqrt(y ** 2 + x ** 2) - major) ** 2 + z ** 2
    return distance <= minor ** 2


@pytest.mark.parametrize("fg,bg", sorted(JORDAN_PAIRS))
@pytest.mark.parametrize(
    "volume,expected",
    [(sealed_shell(), (0, 1)), (solid_torus(), (1, 0))],
    ids=["sealed_shell", "solid_torus"],
)
def test_unambiguous_shapes_agree_across_both_pairs(fg, bg, volume, expected):
    """A shape with no diagonal-only contact has one answer, either convention."""
    b1, b2, _ = compute_component_topology(
        volume, fg_connectivity=fg, bg_connectivity=bg, use_cpu=True
    )
    assert (b1, b2) == expected


@pytest.mark.parametrize(
    "fg,bg,expected",
    [
        # Under a 26-connected background the corner contact joins the interior
        # to the outside, so there is no cavity and no handle. Under a
        # 6-connected background it stays sealed, which is that convention's
        # correct answer - and the foreground Euler characteristic tracks it, so
        # b1 is 0 either way.
        (6, 26, (0, 0)),
        (26, 6, (0, 1)),
    ],
)
def test_the_corner_leak_is_read_consistently(fg, bg, expected):
    b1, b2, _ = compute_component_topology(
        leaky_shell(), fg_connectivity=fg, bg_connectivity=bg, use_cpu=True
    )
    assert (b1, b2) == expected


@pytest.mark.parametrize("fg,bg", [(6, 6), (26, 26), (6, 18), (18, 6), (26, 18)])
def test_non_complementary_pairs_are_refused(fg, bg):
    with pytest.raises(ValueError, match="complementary"):
        compute_component_topology(
            sealed_shell(), fg_connectivity=fg, bg_connectivity=bg, use_cpu=True
        )


def test_betti_numbers_are_never_negative_for_allowed_pairs():
    """(26, 26) used to return b1 = -1 on this shape; no allowed pair may."""
    for fg, bg in sorted(JORDAN_PAIRS):
        b1, b2, _ = compute_component_topology(
            leaky_shell(), fg_connectivity=fg, bg_connectivity=bg, use_cpu=True
        )
        assert b1 >= 0 and b2 >= 0


def connectivity_actions(module):
    """The two connectivity arguments as argparse actions, from a real parse."""
    import argparse
    from unittest import mock

    captured = {}

    def capture(self, *_args, **_kwargs):
        captured["parser"] = self
        raise SystemExit(0)

    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
        with pytest.raises(SystemExit):
            module.parse_args()

    return {
        action.dest: action
        for action in captured["parser"]._actions
        if action.dest in ("fg_connectivity", "bg_connectivity")
    }


def test_the_cli_defaults_are_a_complementary_pair():
    from vesuvius.scripts import detect_holes

    actions = connectivity_actions(detect_holes)
    check_connectivity_pair(
        actions["fg_connectivity"].default, actions["bg_connectivity"].default
    )


def test_every_offered_combination_is_a_complementary_pair():
    """No selectable combination may produce a meaningless count."""
    from vesuvius.scripts import detect_holes

    actions = connectivity_actions(detect_holes)
    for fg in actions["fg_connectivity"].choices:
        for bg in actions["bg_connectivity"].choices:
            if (fg, bg) in JORDAN_PAIRS:
                continue
            # The parser rejects it before any volume is read.
            with pytest.raises(ValueError, match="complementary"):
                check_connectivity_pair(fg, bg)


def test_fg_connectivity_18_is_not_offered():
    """No Euler backend implements 18 in 3D, so it must not be selectable."""
    from vesuvius.scripts import detect_holes

    actions = connectivity_actions(detect_holes)
    assert 18 not in actions["fg_connectivity"].choices
