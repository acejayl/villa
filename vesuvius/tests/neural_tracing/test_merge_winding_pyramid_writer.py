"""Two upsampled tiles sharing a chunk must not erase each other.

OmeZarrPyramidWriter.write_block zero-pads the part of a chunk its block does
not cover, on the documented assumption that "the band is this array's only
content, so no read-modify-write is ever needed". That holds for levels 2..5,
which each write a single block spanning the whole band. It does not hold for
levels 0 and 1, which write one block per upsampled tile.

make_band aligns a band's z_lo to BAND_ALIGN = 32, but a level-0 chunk is
CHUNK = 128. Level-0 tiles start at z_lo + z0 * 4 with z0 a multiple of 128, so
they are 512 apart - chunk aligned relative to each other, but shifted by
z_lo % 128 in absolute terms. When that is non-zero (3 of the 4 possible band
alignments) two z-adjacent tiles land in the same chunk, and the second block
written replaces it with a chunk that is zeros over the first block's rows.

Zero is the invalid code, so the loss is silent: solved winding values come
back as unpopulated.
"""

import numpy as np
import pytest

from vesuvius.neural_tracing.winding_models.merge_winding_field import (
    BAND_ALIGN,
    CHUNK,
    BandGeometry,
    OmeZarrPyramidWriter,
    make_band,
)

FULL_SHAPE = (4096, 512, 512)


def make_writer(tmp_path, z_lo, z_hi):
    return OmeZarrPyramidWriter(
        tmp_path / "field.zarr",
        BandGeometry(FULL_SHAPE, z_lo, z_hi),
        source="test",
        parameters={},
        zstd_level=1,
    )


def read_chunk(writer, level, chunk_index):
    path = writer.path / str(level) / "/".join(str(i) for i in chunk_index)
    raw = writer.compressor.decode(path.read_bytes())
    return np.frombuffer(raw, dtype=np.uint16).reshape(CHUNK, CHUNK, CHUNK)


@pytest.mark.parametrize("z_lo", [0, 32, 64, 96])
def test_adjacent_level0_tiles_both_survive(tmp_path, z_lo):
    """The two blocks a shared chunk sees are disjoint; both must be kept."""
    writer = make_writer(tmp_path / str(z_lo), z_lo, z_lo + 2048)
    tile = 4 * CHUNK  # one ds4 tile upsampled to level 0

    # Two z-adjacent tiles, each a distinct constant so we can tell them apart.
    for index, origin_z in enumerate((z_lo, z_lo + tile)):
        values = np.full((tile, CHUNK, CHUNK), index + 1, dtype=np.uint16)
        writer.write_block(0, (origin_z, 0, 0), values)
    writer._drain()

    # Every level-0 voxel the two tiles covered must still carry its tile's
    # value, whichever chunk it landed in.
    for index, origin_z in enumerate((z_lo, z_lo + tile)):
        for offset in (0, tile // 2, tile - 1):
            z = origin_z + offset
            chunk = read_chunk(writer, 0, (z // CHUNK, 0, 0))
            assert chunk[z % CHUNK, 0, 0] == index + 1, (
                f"z={z} from tile {index} was overwritten "
                f"(z_lo={z_lo}, chunk {z // CHUNK})"
            )


def test_shared_chunk_holds_both_tiles(tmp_path):
    """The straddled chunk itself: bottom rows tile 0, top rows tile 1."""
    z_lo = 32
    writer = make_writer(tmp_path, z_lo, z_lo + 2048)
    tile = 4 * CHUNK
    for index, origin_z in enumerate((z_lo, z_lo + tile)):
        writer.write_block(
            0, (origin_z, 0, 0),
            np.full((tile, CHUNK, CHUNK), index + 1, dtype=np.uint16))
    writer._drain()

    shared = (z_lo + tile) // CHUNK
    chunk = read_chunk(writer, 0, (shared, 0, 0))
    boundary = (z_lo + tile) % CHUNK
    assert np.all(chunk[:boundary] == 1), "tile 0's rows were erased"
    assert np.all(chunk[boundary:] == 2), "tile 1's rows were erased"


def test_make_band_can_produce_a_non_chunk_aligned_origin():
    """Why the collision is reachable: band alignment is coarser than a chunk."""
    assert BAND_ALIGN < CHUNK
    band = make_band(FULL_SHAPE, (1000, 1200), margin=0)
    assert band.z_lo % BAND_ALIGN == 0
    assert band.z_lo % CHUNK != 0


def test_a_fresh_chunk_is_not_merged_with_a_previous_run(tmp_path):
    """Merging is scoped to this writer, so a re-run still starts clean."""
    path = tmp_path / "field.zarr"
    band = BandGeometry(FULL_SHAPE, 0, 2048)
    first = OmeZarrPyramidWriter(path, band, "test", {}, 1)
    first.write_block(
        0, (0, 0, 0), np.full((CHUNK, CHUNK, CHUNK), 7, dtype=np.uint16))
    first._drain()

    second = OmeZarrPyramidWriter(path, band, "test", {}, 1)
    values = np.full((CHUNK, CHUNK, CHUNK), 9, dtype=np.uint16)
    values[:10] = 0
    second.write_block(0, (0, 0, 0), values)
    second._drain()

    chunk = read_chunk(second, 0, (0, 0, 0))
    assert np.all(chunk[:10] == 0), "stale data from an earlier run was kept"
    assert np.all(chunk[10:] == 9)
