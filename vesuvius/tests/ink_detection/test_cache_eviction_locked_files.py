"""Cache eviction must survive an entry another process is reading.

`_evict_to_watermark` caught only FileNotFoundError around `path.unlink()`, and
`_cache_snapshot` caught only FileNotFoundError around `path.stat()`. On POSIX
that is nearly enough. On Windows it is not: unlinking or statting a file that
another process holds open raises

    PermissionError: [WinError 32] The process cannot access the file because
    it is being used by another process

and this cache is explicitly shared between worker processes, so one worker
reading an entry while another sweeps is the ordinary case, not an error.

The exception propagated out of the sweep, so a routine race aborted whatever
triggered it.

Skipping a locked entry is the right behaviour - it stays on disk and a later
sweep gets it - so its size has to stay counted, which is what the size
assertions below check.
"""

import pytest

from vesuvius.ink_detection.volume_io import _cache_snapshot, _evict_to_watermark

SIZE = 4096


def make_entry(directory, name, size=SIZE):
    path = directory / name
    path.write_bytes(b"x" * size)
    return path


def test_eviction_survives_an_entry_held_open(tmp_path):
    """The regression: this raised PermissionError on Windows."""
    entry = make_entry(tmp_path, "held.bin")
    snapshot = _cache_snapshot(tmp_path)

    with entry.open("rb"):
        remaining = _evict_to_watermark(snapshot, max_bytes=1)

    assert entry.exists(), "a held entry should be left for a later sweep"
    assert remaining == SIZE, "a file still on disk must stay counted"


def test_an_unheld_entry_is_still_evicted(tmp_path):
    make_entry(tmp_path, "free.bin")
    snapshot = _cache_snapshot(tmp_path)

    remaining = _evict_to_watermark(snapshot, max_bytes=1)

    assert not (tmp_path / "free.bin").exists()
    assert remaining == 0


def test_a_held_entry_does_not_stop_the_others_being_evicted(tmp_path):
    """One locked file must not abort the whole sweep."""
    held = make_entry(tmp_path, "0_held.bin")
    free_a = make_entry(tmp_path, "1_free.bin")
    free_b = make_entry(tmp_path, "2_free.bin")
    snapshot = _cache_snapshot(tmp_path)

    with held.open("rb"):
        _evict_to_watermark(snapshot, max_bytes=1)

    assert held.exists()
    assert not free_a.exists()
    assert not free_b.exists()


def test_an_already_deleted_entry_is_not_counted(tmp_path):
    """FileNotFoundError still means the space is genuinely free."""
    entry = make_entry(tmp_path, "gone.bin")
    snapshot = _cache_snapshot(tmp_path)
    entry.unlink()

    assert _evict_to_watermark(snapshot, max_bytes=1) == 0


def test_snapshot_skips_a_partial_entry(tmp_path):
    make_entry(tmp_path, "done.bin")
    make_entry(tmp_path, "writing.partial")

    names = [path.name for _, _, path in _cache_snapshot(tmp_path)]

    assert names == ["done.bin"]


def test_eviction_stops_at_the_watermark(tmp_path):
    for index in range(4):
        make_entry(tmp_path, f"{index}.bin")
    snapshot = _cache_snapshot(tmp_path)

    remaining = _evict_to_watermark(snapshot, max_bytes=4 * SIZE)

    # 0.9 * max_bytes is the target, so one entry has to go and no more.
    assert remaining == pytest.approx(3 * SIZE)
    assert sum(1 for p in tmp_path.iterdir() if p.is_file()) == 3

