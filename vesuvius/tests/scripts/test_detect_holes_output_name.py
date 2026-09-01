"""--hole-mask-name has to be able to name the output.

In --subfolders mode the output filename was chosen by

    if args.hole_mask_suffix:
        out_name = f"hole_mask_{args.hole_mask_suffix}.tif"
    else:
        out_name = args.hole_mask_name

and just above it hole_mask_suffix was filled in from --label-glob whenever it
was unset. _suffix_from_glob never returns an empty string - it falls back to
"label" - so the suffix was always truthy, the else branch was unreachable, and
--hole-mask-name did nothing at all. Both flags are documented as "used with
--subfolders", and that is the only place the name is read.

Precedence: an explicit --hole-mask-suffix, then an explicit --hole-mask-name,
then the suffix derived from the label glob, which is what an invocation with
neither has always produced.
"""

import pytest

from vesuvius.scripts.detect_holes import _suffix_from_glob, parse_args

DEFAULT_GLOB = "*_surf.tif"


def resolve(suffix, name, glob=DEFAULT_GLOB):
    from vesuvius.scripts.detect_holes import resolve_hole_mask_name

    return resolve_hole_mask_name(suffix, name, glob)


def test_an_explicit_name_is_used():
    assert resolve(None, "my_holes.tif") == "my_holes.tif"


def test_an_explicit_suffix_wins_over_a_name():
    assert resolve("v2", "my_holes.tif") == "hole_mask_v2.tif"


def test_with_neither_the_suffix_comes_from_the_label_glob():
    """Unchanged from before: what a default invocation has always produced."""
    assert resolve(None, None) == "hole_mask_surf.tif"


def test_a_custom_label_glob_still_drives_the_derived_suffix():
    assert resolve(None, None, "*_2d_hole_fill.tif") == "hole_mask_2d_hole_fill.tif"


def test_the_name_flag_defaults_to_unset(monkeypatch):
    """It has to, or an explicit name is indistinguishable from the default."""
    monkeypatch.setattr(
        "sys.argv", ["detect_holes.py", "--in-dir", "somewhere", "--subfolders"]
    )
    args = parse_args()

    assert args.hole_mask_name is None
    assert args.hole_mask_suffix is None


def test_the_flag_is_read_through_the_parser(monkeypatch):
    """End to end from the command line to the filename."""
    monkeypatch.setattr("sys.argv", [
        "detect_holes.py", "--in-dir", "somewhere", "--subfolders",
        "--hole-mask-name", "custom.tif",
    ])
    args = parse_args()

    assert resolve(
        args.hole_mask_suffix, args.hole_mask_name, args.label_glob
    ) == "custom.tif"


@pytest.mark.parametrize(
    "glob,expected", [("*_surf.tif", "surf"), ("label.tif", "label"), ("*", "label")]
)
def test_suffix_from_glob_never_returns_empty(glob, expected):
    """Why the else branch was unreachable."""
    assert _suffix_from_glob(glob) == expected
