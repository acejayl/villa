"""The chunk-occupancy index must work for nested (dimension_separator="/") arrays.

_list_chunks_local built nested-layout keys straight from os.scandir paths, so on
Windows they came back backslash-separated ("0\\1\\0"). The caller splits those
on the zarr dimension separator "/", producing one part instead of `rank`, and
every key was dropped by a bare `continue` - not even counted as skipped. The
index therefore never parsed a single chunk, build_chunk_occupancy always warned
"No parseable chunk files found" and returned None, and the sparse-volume patch
pre-filter in data/vc_dataset.py was permanently inert.

The flat (".") layout is unaffected because it appends entry.name, and the S3
branch is unaffected because S3 keys always use "/". The existing suite only
covers ".", and CI is ubuntu-only, so nothing caught it.

test_nested_layout_is_parsed fails on Windows against the previous
implementation.
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from vesuvius.data.zarr_chunk_index import _list_chunks_local, build_chunk_occupancy

SHAPE = (4, 4, 4)
CHUNKS = (2, 2, 2)


def _make_array(root: Path, separator: str) -> Path:
    array = root / "volume.zarr" / "0"
    array.mkdir(parents=True)
    (array / ".zarray").write_text(
        json.dumps({
            "zarr_format": 2, "shape": list(SHAPE), "chunks": list(CHUNKS),
            "dtype": "|u1", "compressor": None, "fill_value": 0, "order": "C",
            "filters": None, "dimension_separator": separator,
        }),
        encoding="ascii",
    )
    # four occupied chunks: (z, y, 0) for z, y in {0, 1}
    for z in range(2):
        for y in range(2):
            if separator == "/":
                d = array / str(z) / str(y)
                d.mkdir(parents=True, exist_ok=True)
                (d / "0").write_bytes(b"non-empty")
            else:
                (array / f"{z}.{y}.0").write_bytes(b"non-empty")
    return array


def test_nested_keys_use_forward_slashes(tmp_path):
    array = _make_array(tmp_path, "/")
    keys = _list_chunks_local(str(array), "0/", "/")

    assert keys, "no chunk files listed at all"
    for key in keys:
        assert os.sep not in key or os.sep == "/", f"key uses an OS separator: {key!r}"
        assert len(key.split("/")) == len(SHAPE), f"key does not split to rank: {key!r}"


@pytest.mark.parametrize("separator", ["/", "."], ids=["nested", "flat"])
def test_occupancy_is_built_for_both_layouts(tmp_path, monkeypatch, separator):
    monkeypatch.setenv("VESUVIUS_CACHE_DIR", str(tmp_path / "cache"))
    array = _make_array(tmp_path, separator)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        occupancy = build_chunk_occupancy(
            str(array), chunks=CHUNKS, shape=SHAPE, use_cache=False
        )

    messages = [str(w.message) for w in caught]
    assert occupancy is not None, f"no occupancy built; warnings: {messages}"
    assert occupancy.shape == (2, 2, 2)
    assert int(occupancy.sum()) == 4, f"expected 4 occupied chunks, got {int(occupancy.sum())}"
    assert not any("No parseable chunk files" in m for m in messages), messages


def test_nested_and_flat_agree(tmp_path, monkeypatch):
    monkeypatch.setenv("VESUVIUS_CACHE_DIR", str(tmp_path / "cache"))
    nested = build_chunk_occupancy(
        str(_make_array(tmp_path / "n", "/")), chunks=CHUNKS, shape=SHAPE, use_cache=False
    )
    flat = build_chunk_occupancy(
        str(_make_array(tmp_path / "f", ".")), chunks=CHUNKS, shape=SHAPE, use_cache=False
    )
    assert nested is not None and flat is not None
    np.testing.assert_array_equal(nested, flat)
