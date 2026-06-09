"""Unit tests for src.anchor.threshold — calibration T, per-FIT δ / T'."""

from __future__ import annotations

import math

import pytest

from src.anchor.threshold import (
    CalibrationResult,
    cosine_distance,
    derive_threshold_prime,
    loo_nearest_distances,
    quantile,
    span_loo_distances,
    span_to_chunk_distances,
)


# ---------------------------------------------------------------------
# Pure-helper regression (unchanged behaviour from the previous module)
# ---------------------------------------------------------------------


def test_cosine_distance_identical_vectors():
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


def test_cosine_distance_orthogonal():
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_cosine_distance_zero_magnitude_returns_one():
    assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_quantile_p90_interpolation_matches_numpy_linear():
    # For sorted values [0..14] (N=15), p90 position = 0.9 * 14 = 12.6
    # -> value = 12 + 0.6 * (13 - 12) = 12.6
    vals = [float(i) for i in range(15)]
    assert quantile(vals, 0.90) == pytest.approx(12.6)


# ---------------------------------------------------------------------
# span_loo_distances / span_to_chunk_distances
# ---------------------------------------------------------------------


def test_span_loo_distances_is_loo_nearest_distances():
    # Alias contract: same function, renamed for callsite clarity.
    vecs = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    assert span_loo_distances(vecs) == loo_nearest_distances(vecs)


def test_span_to_chunk_distances_zips_pairwise():
    spans = [[1.0, 0.0], [0.0, 1.0]]
    chunks = [[1.0, 0.0], [0.0, 1.0]]
    assert span_to_chunk_distances(spans, chunks) == pytest.approx([0.0, 0.0])


def test_span_to_chunk_distances_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        span_to_chunk_distances([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])


# ---------------------------------------------------------------------
# derive_threshold_prime
# ---------------------------------------------------------------------


def test_derive_threshold_prime_fields_and_math():
    # Construct N=4 span vectors with known nearest-other distances
    # [0.1, 0.1, 0.2, 0.2] -> sorted [0.1, 0.1, 0.2, 0.2], p90 at
    # position 2.7 -> 0.2 + 0.7 * 0 = 0.2. Span-to-chunk pairs
    # [0.01, 0.02, 0.03, 0.04] become per-FIT δ_i directly — no median.
    # Each T'_i = T + δ_i.
    span_vecs = _vecs_with_nearest(nearest_distances=[0.1, 0.1, 0.2, 0.2])
    chunk_vecs = _vecs_at_distances(span_vecs, distances=[0.01, 0.02, 0.03, 0.04])

    r = derive_threshold_prime(span_vecs, chunk_vecs)

    assert isinstance(r, CalibrationResult)
    assert r.n_fit == 4
    assert r.T == pytest.approx(0.2, abs=1e-9)
    assert r.deltas == pytest.approx([0.01, 0.02, 0.03, 0.04])
    assert r.T_primes == pytest.approx([0.21, 0.22, 0.23, 0.24])
    # T_prime_out = T + min(δ) = 0.2 + 0.01 = 0.21.
    assert r.T_prime_out == pytest.approx(0.21, abs=1e-9)
    assert sorted(r.span_loo_distances) == pytest.approx([0.1, 0.1, 0.2, 0.2])
    # deltas[i] == raw span-to-own-chunk distance; no quantile collapse.
    assert not hasattr(r, "delta")
    assert not hasattr(r, "T_prime")
    assert not hasattr(r, "span_to_chunk_distances")


def test_derive_threshold_prime_T_prime_out_is_T_plus_min_delta():
    # Skew one δ so min ≠ median ≠ mean; prove T_prime_out follows min.
    # deltas = [0.01, 0.02, 0.03, 0.50]
    #   min    = 0.01
    #   median = 0.025
    #   mean   = 0.14
    span_vecs = _vecs_with_nearest(nearest_distances=[0.1, 0.1, 0.2, 0.2])
    chunk_vecs = _vecs_at_distances(span_vecs, distances=[0.01, 0.02, 0.03, 0.50])

    r = derive_threshold_prime(span_vecs, chunk_vecs)

    # Min-not-median/mean: if this used median it would be T + 0.025;
    # if mean, T + 0.14.
    assert r.T_prime_out == pytest.approx(r.T + 0.01, abs=1e-9)


def test_derive_threshold_prime_requires_minimum_two_fits():
    with pytest.raises(ValueError, match="at least 2"):
        derive_threshold_prime(
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        )


def test_derive_threshold_prime_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        derive_threshold_prime(
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0]],
        )


# ---------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------


def _vecs_with_nearest(nearest_distances: list[float]) -> list[list[float]]:
    """Build N 2-D unit vectors whose mutual LOO-nearest distances
    are exactly ``nearest_distances``.

    Construction: place N points on the unit circle at angles chosen
    so each neighbour pair has the requested cosine distance.
    """
    # Simple construction: pair up points (i, i+1) so (2k, 2k+1) are
    # close by distances[2k]. Works for even N; we only use N=4 here.
    assert len(nearest_distances) % 2 == 0, "Use an even N in fixtures."
    vecs: list[list[float]] = []
    # Spread pairs far apart so intra-pair is the nearest neighbour.
    base_angles = [0.0, math.pi / 2]  # two well-separated anchors
    for pair_idx, base in enumerate(base_angles):
        # cos(theta) = 1 - d  => theta = acos(1 - d)
        d = nearest_distances[2 * pair_idx]
        theta = math.acos(1.0 - d)
        vecs.append([math.cos(base), math.sin(base)])
        vecs.append([math.cos(base + theta), math.sin(base + theta)])
    return vecs


def _vecs_at_distances(
    anchors: list[list[float]], distances: list[float]
) -> list[list[float]]:
    """Return a list of vectors such that ``cosine_distance(anchors[i],
    result[i]) == distances[i]``."""
    out: list[list[float]] = []
    for a, d in zip(anchors, distances):
        # cos(theta) = 1 - d
        theta = math.acos(1.0 - d)
        # Rotate ``a`` by theta in 2-D.
        ax, ay = a
        rx = ax * math.cos(theta) - ay * math.sin(theta)
        ry = ax * math.sin(theta) + ay * math.cos(theta)
        out.append([rx, ry])
    return out
