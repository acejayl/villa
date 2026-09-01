"""_scan_url must strip the level directory on Windows paths too.

The sibling multiscale level was derived with POSIX string surgery:

    base_url = self.labels_zarr_url.rstrip("/")
    last = base_url.rsplit("/", 1)[-1]
    parent = base_url.rsplit("/", 1)[0] if last.isdigit() else base_url
    return f"{parent}/{level}"

labels_zarr_url is documented and shipped as a plain local path pointing
straight at a level directory - the configs say
"/ephemeral/fibers_labels/s1a-fibers-230125-ome.zarr/0". On Windows the same
tree is "D:\\fibers_labels\\s1a-fibers-230125-ome.zarr\\0", where rsplit("/")
finds no separator, so last.isdigit() is False and the inline comment's stated
guarantee - "Strip trailing ``/<digit>`` so ``<base>/0`` -> ``<base>``" - was
silently not kept. The level got appended to the level directory.

The backslash cases fail against the previous implementation; the POSIX and
URL cases pass either way.
"""

import pytest

from vesuvius.models.datasets.cross_frame_dataset import CrossFrameZarrDataset


def _scan_url(labels_zarr_url, level):
    """Call the method without constructing the dataset (which opens zarr)."""
    stub = object.__new__(CrossFrameZarrDataset)
    stub.labels_zarr_url = labels_zarr_url
    return CrossFrameZarrDataset._scan_url(stub, level)


@pytest.mark.parametrize(
    "url,level,expected",
    [
        # POSIX, as the shipped configs spell it - already worked
        ("/data/s1a-fibers-ome.zarr/0", 2, "/data/s1a-fibers-ome.zarr/2"),
        ("/data/s1a-fibers-ome.zarr/0/", 2, "/data/s1a-fibers-ome.zarr/2"),
        ("/data/s1a-fibers-ome.zarr", 2, "/data/s1a-fibers-ome.zarr/2"),
        # a real URL - must keep forward slashes
        ("s3://bucket/labels.zarr/0", 3, "s3://bucket/labels.zarr/3"),
        # Windows, the same tree - previously produced "...zarr\\0/2"
        (r"D:\labels\s1a-fibers-ome.zarr\0", 2, r"D:\labels\s1a-fibers-ome.zarr\2"),
        (r"D:\labels\s1a-fibers-ome.zarr\0" + "\\", 2, r"D:\labels\s1a-fibers-ome.zarr\2"),
        (r"D:\labels\s1a-fibers-ome.zarr", 2, r"D:\labels\s1a-fibers-ome.zarr\2"),
    ],
    ids=["posix", "posix_trailing_slash", "posix_no_level", "s3_url",
         "windows", "windows_trailing_sep", "windows_no_level"],
)
def test_scan_url_strips_the_level_directory(url, level, expected):
    assert _scan_url(url, level) == expected


def test_a_non_numeric_final_component_is_not_stripped():
    """Only a trailing digit is a level; a named directory must survive."""
    assert _scan_url("/data/labels.zarr/main", 1) == "/data/labels.zarr/main/1"
    assert _scan_url(r"D:\data\labels.zarr\main", 1) == r"D:\data\labels.zarr\main\1"
