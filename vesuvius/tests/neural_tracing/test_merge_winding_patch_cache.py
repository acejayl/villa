"""The cached patch list has to be keyed on the band it was filtered by.

load_patch_metas returns the patches whose z bbox overlaps the solve band, and
caches that list in the scratch directory. The cache key was patches_dir alone.

Scratch defaults to the output path's stem and can be shared explicitly with
--scratch-dir, which the module docstring encourages ("every pass persists its
product to --scratch-dir and is skipped"). So re-running with a different
--z-range reused the previous band's list: missing the patches the new band
needs, and carrying patches that do not reach it.

Nothing downstream catches it. rasterize_patches does stamp the band, but it
stamps len(names), not the names, so it happily rebuilds its grid from the
wrong list.
"""

import json

from vesuvius.neural_tracing.winding_models.merge_winding_field import (
    BandGeometry,
    load_patch_metas,
)

FULL_SHAPE = (4096, 512, 512)


def make_patches(tmp_path, spans):
    """spans: {name: (z_min, z_max)} -> a patches dir with those bboxes."""
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir(parents=True)
    for name, (z_min, z_max) in spans.items():
        directory = patches_dir / name
        directory.mkdir()
        (directory / "meta.json").write_text(
            json.dumps({"bbox": [[0, 0, z_min], [10, 10, z_max]]}),
            encoding="utf-8",
        )
    return patches_dir


def band(z_lo, z_hi):
    return BandGeometry(FULL_SHAPE, z_lo, z_hi)


SPANS = {
    "low": (0, 500),
    "middle": (1000, 1500),
    "high": (2000, 2500),
}


def test_a_second_band_does_not_reuse_the_first_bands_list(tmp_path):
    patches_dir = make_patches(tmp_path, SPANS)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    first = load_patch_metas(patches_dir, band(0, 800), scratch)
    second = load_patch_metas(patches_dir, band(1900, 2600), scratch)

    assert first == ["low"]
    assert second == ["high"], (
        f"got {second} - the first band's cached list was reused"
    )


def test_the_same_band_still_hits_the_cache(tmp_path):
    patches_dir = make_patches(tmp_path, SPANS)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    first = load_patch_metas(patches_dir, band(900, 1600), scratch)

    # Remove the source directories: only a cache hit can answer now.
    for name in SPANS:
        (patches_dir / name / "meta.json").unlink()

    assert load_patch_metas(patches_dir, band(900, 1600), scratch) == first
    assert first == ["middle"]


def test_a_different_patches_dir_still_misses(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    one = make_patches(tmp_path / "one", {"low": (0, 500)})
    two = make_patches(tmp_path / "two", {"high": (2000, 2500)})

    assert load_patch_metas(one, band(0, 3000), scratch) == ["low"]
    assert load_patch_metas(two, band(0, 3000), scratch) == ["high"]


def test_the_cache_records_the_band_it_was_built_for(tmp_path):
    patches_dir = make_patches(tmp_path, SPANS)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    load_patch_metas(patches_dir, band(1000, 1600), scratch)
    cached = json.loads((scratch / "patch_metas.json").read_text())

    assert cached["band"] == [1000, 1600]
    assert cached["patches_dir"] == str(patches_dir)
    assert cached["names"] == ["middle"]
