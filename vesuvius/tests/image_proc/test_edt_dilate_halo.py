"""edt-dilate must dilate across block boundaries and not fake a border per block.

Two independent defects, both silent:

1. Each block was read on exactly its own bounds and blocks with no foreground
   returned early, so dilation could never cross a block face. A sheet at z=14
   dilated by 5 marked z=9..15 instead of z=9..19, losing 4096 of 11264 voxels.
2. black_border was passed straight to edt() on the *inverted* mask, so
   black_border=True told edt that everything outside each block was
   foreground. Every processed block grew a false hollow-box halo: with one
   seed voxel and distance 5, 3583 voxels were marked that should not be,
   including (0, 0, 0), which is 18.8 voxels from the nearest foreground.

These tests fail against the previous implementation.
"""

import numpy as np
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks.edt_dilate import EdtDilateConfig, EdtDilateTask

SIZE = 32
BLOCK = (16, 16, 16)
DISTANCE = 5.0


def _dilate(tmp_path, volume, black_border, distance=DISTANCE, block=BLOCK):
    src = tmp_path / "in.zarr"
    out = tmp_path / "out.zarr"
    group = zarr.open_group(str(src), mode="w")
    if hasattr(group, "create_dataset"):          # zarr 2
        group.create_dataset("0", data=volume, chunks=block)
    else:                                          # zarr 3 removed it
        array = group.create_array(
            "0", shape=volume.shape, chunks=block, dtype=volume.dtype)
        array[:] = volume
    EdtDilateTask(EdtDilateConfig(
        input_zarr=str(src), output_zarr=str(out), num_workers=1,
        distance=distance, chunk_size=block, black_border=black_border,
        resolution="0",
    )).run()
    return np.asarray(zarr.open(str(out), mode="r")["0"][:]).astype(bool)


def _ball(seed, radius, size=SIZE):
    zz, yy, xx = np.ogrid[:size, :size, :size]
    d = np.sqrt((zz - seed[0]) ** 2 + (yy - seed[1]) ** 2 + (xx - seed[2]) ** 2)
    return d <= radius


def test_seed_on_a_block_boundary_dilates_into_the_next_block(tmp_path):
    seed = (15, 8, 8)  # z=15 is the last plane of block 0
    volume = np.zeros((SIZE, SIZE, SIZE), np.uint8)
    volume[seed] = 1

    got = _dilate(tmp_path, volume, black_border=False)
    want = _ball(seed, DISTANCE)

    assert int((want & ~got).sum()) == 0, "dilation was truncated at the block face"
    assert int((got & ~want).sum()) == 0, "voxels marked that are out of range"

    z_present = np.where(got.any(axis=(1, 2)))[0]
    assert (int(z_present.min()), int(z_present.max())) == (10, 20)
    assert got[16:].sum() > 0, "nothing reached the neighbouring block"


def test_sheet_dilates_symmetrically_across_blocks(tmp_path):
    volume = np.zeros((SIZE, SIZE, SIZE), np.uint8)
    volume[14, :, :] = 1

    got = _dilate(tmp_path, volume, black_border=False)

    expected_slices = [z for z in range(SIZE) if abs(z - 14) <= DISTANCE]
    assert [int(z) for z in np.where(got.any(axis=(1, 2)))[0]] == expected_slices


def test_no_false_border_on_interior_block_faces(tmp_path):
    """black_border must not manufacture foreground at a block face."""
    seed = (15, 8, 8)
    volume = np.zeros((SIZE, SIZE, SIZE), np.uint8)
    volume[seed] = 1

    got = _dilate(tmp_path, volume, black_border=False)

    assert not got[0, 0, 0], "corner marked though it is 18.8 voxels from any foreground"
    # the plane just inside the far side of block 0 is >5 voxels from the seed
    assert not got[0, :, :].any(), "the z=0 plane of block 0 was falsely marked"


def test_black_border_applies_to_the_volume_edge_not_every_block(tmp_path):
    """With black_border=True only the outer boundary of the volume seeds dilation."""
    volume = np.zeros((SIZE, SIZE, SIZE), np.uint8)
    volume[15, 8, 8] = 1

    got = _dilate(tmp_path, volume, black_border=True)

    # the outer shell is marked ...
    assert got[0, :, :].all(), "volume boundary did not seed dilation"
    # ... but the deep interior, far from both the seed and every face, is not
    interior = got[12:20, 12:20, 12:20]
    assert not interior.all(), "interior flooded; border was applied per block"
