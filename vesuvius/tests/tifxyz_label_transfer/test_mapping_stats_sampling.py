"""Reported distance percentiles must describe the whole surface.

MappingStats.add is called once per transfer tile, from a thread pool. It kept
distances for percentiles until a sample budget was spent and then dropped
every later tile outright, so distance_p50/p95/p99 described whichever tiles
happened to finish first - a subset that is neither representative of the
surface nor stable between runs.

A surface whose mapping quality varies across it is exactly the case those
percentiles exist to report on, and exactly the case the truncation got wrong.
"""

import numpy as np
import pytest

from vesuvius.tifxyz_label_transfer.core import MappingStats

TILES = 10
PER_TILE = 1000
LIMIT = 1000


def tile_distances(index):
    """Tile k covers distances [k, k+1), so the surface spans [0, TILES)."""
    return np.linspace(
        index, index + 1, PER_TILE, endpoint=False, dtype=np.float32
    )


def feed(order, limit=LIMIT):
    stats = MappingStats()
    for index in order:
        stats.add(
            target_pixels=PER_TILE,
            target_surface_valid=PER_TILE,
            distances=tile_distances(index),
            sample_limit=limit,
        )
    return stats.as_dict()


ASCENDING = list(range(TILES))
DESCENDING = list(reversed(range(TILES)))
INTERLEAVED = [i for pair in zip(range(5), range(9, 4, -1)) for i in pair]


@pytest.mark.parametrize(
    "order",
    [ASCENDING, DESCENDING, INTERLEAVED],
    ids=["ascending", "descending", "interleaved"],
)
def test_percentiles_see_the_whole_stream(order):
    """The budget is a tenth of the stream, so only a sample can be kept - but
    it has to be a sample of all of it."""
    result = feed(order)

    assert result["distance_p50"] == pytest.approx(5.0, abs=0.4)
    assert result["distance_p95"] == pytest.approx(9.5, abs=0.4)
    assert result["distance_p99"] == pytest.approx(9.9, abs=0.4)

    # The mean is accumulated exactly, never sampled, so it is the ground truth
    # the percentiles have to be consistent with.
    assert result["distance_mean"] == pytest.approx(5.0, abs=0.01)


def test_tile_completion_order_does_not_change_the_report():
    """Tiles finish in thread-pool order, so the report must not depend on it."""
    reports = [feed(order) for order in (ASCENDING, DESCENDING, INTERLEAVED)]
    for key in ("distance_p50", "distance_p95", "distance_p99"):
        values = [report[key] for report in reports]
        assert max(values) - min(values) < 0.6, (
            f"{key} varied with tile order: {values}"
        )


def test_the_report_is_reproducible():
    assert feed(ASCENDING) == feed(ASCENDING)


def test_a_stream_under_the_budget_is_kept_exactly():
    stats = MappingStats()
    distances = np.arange(100, dtype=np.float32)
    stats.add(target_pixels=100, target_surface_valid=100, distances=distances)
    result = stats.as_dict()

    assert result["distance_p50"] == pytest.approx(np.percentile(distances, 50))
    assert result["distance_p95"] == pytest.approx(np.percentile(distances, 95))
    assert result["distance_min"] == pytest.approx(0.0)
    assert result["distance_max"] == pytest.approx(99.0)
    assert result["mapped_pixels"] == 100


def test_min_max_and_mean_cover_every_tile_regardless_of_budget():
    """Even the tiles the sample dropped are in the exact statistics."""
    result = feed(ASCENDING, limit=1)

    assert result["distance_min"] == pytest.approx(0.0)
    assert result["distance_max"] == pytest.approx(10.0, abs=0.01)
    assert result["mapped_pixels"] == TILES * PER_TILE


def test_non_finite_distances_are_excluded():
    stats = MappingStats()
    distances = np.array([1.0, np.inf, 3.0, np.nan], dtype=np.float32)
    stats.add(target_pixels=4, target_surface_valid=4, distances=distances)
    result = stats.as_dict()

    assert result["mapped_pixels"] == 2
    assert result["distance_mean"] == pytest.approx(2.0)
