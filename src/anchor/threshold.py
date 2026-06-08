"""Threshold derivation — T and per-FIT δ / T' for span-anchored retrieval.

For N FIT spans and their N parent chunks, we compute:

* ``T`` — p90 of each span's leave-one-out nearest-neighbour cosine
  distance to *another* span. Captures the concept's internal spread
  in span-space. Session-scalar.
* ``δ_i`` — ``cosine_distance(span_i, chunk_i)``. Per-FIT bridge from
  the span manifold to the anchor's own chunk. Raw, no aggregation —
  each anchor carries its own offset.
* ``T'_i = T + δ_i`` — per-FIT pass threshold. Used by the LOO gate
  and by the main pass under ``radius_scheme = per_fit``.
* ``T'_out = T + min(δ)`` — session-wide pass threshold used by the
  main pass under ``radius_scheme = decoupled``. ``min`` is the
  reference-session default (see issue #20); it matches the tightest
  natural span-chunk offset in the cohort and has been shown to hold
  cohort consistency under the widest range of anchor drifts.

Quantile for T is hard-coded at 0.90 (an intentional non-tunable).
δ_i is the raw distance; aggregation to ``T'_out`` is a single
``min`` over the post-quality-gate cohort.

Pure Python, no numpy. N is at most a few dozen so the O(N²) pairwise
matrix is trivial.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_T_QUANTILE = 0.90


@dataclass(frozen=True)
class CalibrationResult:
    """Output of :func:`derive_threshold_prime`.

    ``deltas``, ``T_primes``, and ``span_loo_distances`` are parallel
    to the input FIT list — the i-th entry corresponds to FIT i.

    ``T_primes`` are the per-FIT thresholds used by the LOO recovery
    gate regardless of ``radius_scheme``. ``T_prime_out = T + min(δ)``
    is the session-wide cap used by the main pass under the
    ``decoupled`` scheme; the ``per_fit`` scheme uses ``T_primes``
    for the main pass instead.
    """

    T: float
    deltas: list[float]
    T_primes: list[float]
    T_prime_out: float
    span_loo_distances: list[float]
    n_fit: int


def cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine *distance* in [0, 2]: ``1 - cos_sim(a, b)``.

    Returns 1.0 when either vector has zero magnitude — treats undefined
    similarity as "as far as orthogonal" so a degenerate vector neither
    pulls the threshold toward zero nor blows up downstream stats.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dim mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 1.0
    cos = dot / (math.sqrt(na) * math.sqrt(nb))
    if cos > 1.0:
        cos = 1.0
    elif cos < -1.0:
        cos = -1.0
    return 1.0 - cos


def pairwise_distances(vectors: list[list[float]]) -> list[list[float]]:
    """Symmetric N×N cosine-distance matrix; diagonal is 0.0."""
    n = len(vectors)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance(vectors[i], vectors[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def loo_nearest_distances(vectors: list[list[float]]) -> list[float]:
    """For each vector, the cosine distance to its closest *other* vector."""
    n = len(vectors)
    if n < 2:
        raise ValueError(
            f"Need at least 2 vectors to derive a leave-one-out "
            f"nearest-neighbour distance; got {n}"
        )
    matrix = pairwise_distances(vectors)
    out: list[float] = []
    for i in range(n):
        nearest = min(matrix[i][j] for j in range(n) if j != i)
        out.append(nearest)
    return out


# Public alias for callsite clarity — "loo in span-space" reads better
# at the calibration call than the generic helper name.
span_loo_distances = loo_nearest_distances


def span_to_chunk_distances(
    span_vectors: list[list[float]],
    chunk_vectors: list[list[float]],
) -> list[float]:
    """Pairwise cosine distance between each FIT's span and its own chunk."""
    if len(span_vectors) != len(chunk_vectors):
        raise ValueError(
            "span_vectors and chunk_vectors must have the same length; "
            f"got {len(span_vectors)} and {len(chunk_vectors)}"
        )
    return [cosine_distance(s, c) for s, c in zip(span_vectors, chunk_vectors)]


def quantile(values: list[float], q: float) -> float:
    """Inclusive linear-interpolation quantile.

    Matches NumPy's default ``"linear"`` interpolation.
    """
    if not values:
        raise ValueError("Cannot compute quantile of empty list")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1]; got {q}")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def derive_threshold_prime(
    span_vectors: list[list[float]],
    chunk_vectors: list[list[float]],
) -> CalibrationResult:
    """Compute per-FIT pass thresholds T'_i = T + δ_i."""
    if len(span_vectors) != len(chunk_vectors):
        raise ValueError(
            "span_vectors and chunk_vectors must have the same length; "
            f"got {len(span_vectors)} and {len(chunk_vectors)}"
        )
    if len(span_vectors) < 2:
        raise ValueError(
            f"Need at least 2 FITs to derive a threshold; got {len(span_vectors)}"
        )

    span_loo = span_loo_distances(span_vectors)
    deltas = span_to_chunk_distances(span_vectors, chunk_vectors)

    T = quantile(span_loo, _T_QUANTILE)
    T_primes = [T + d for d in deltas]
    T_prime_out = T + min(deltas)

    return CalibrationResult(
        T=T,
        deltas=list(deltas),
        T_primes=T_primes,
        T_prime_out=T_prime_out,
        span_loo_distances=span_loo,
        n_fit=len(span_vectors),
    )


def distance_summary(values: list[float]) -> dict[str, float]:
    """Compact stats block for the meta sidecar."""
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "max": max(values),
    }
