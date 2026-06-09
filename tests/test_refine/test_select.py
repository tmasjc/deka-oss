"""Unit tests for src.refine.select.select_diverse."""

from __future__ import annotations

import math

import pytest

from src.refine.select import select_diverse


def test_returns_input_unchanged_when_within_cap():
    items = ["a", "b"]
    embeds = [[1.0, 0.0], [0.0, 1.0]]
    assert select_diverse(items, embeds, k=5) == ["a", "b"]


def test_picks_distinct_cluster_centers_over_duplicates():
    """Four cluster centers + many duplicates. Asking for k=4 must
    return one item per cluster — duplicates have zero distance to a
    selected member and lose to the next cluster every time."""

    centers = [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ]
    items: list[str] = []
    embeds: list[list[float]] = []
    for label, center in zip("ABCD", centers):
        # 3 copies per cluster: center + two near-duplicates
        items.extend([f"{label}0", f"{label}1", f"{label}2"])
        embeds.extend(
            [
                center,
                [center[0] + 1e-6, center[1]],
                [center[0], center[1] + 1e-6],
            ]
        )

    selected = select_diverse(items, embeds, k=4)
    # Exactly one representative from each cluster.
    clusters = {s[0] for s in selected}
    assert clusters == {"A", "B", "C", "D"}, selected


def test_seed_is_first_item():
    """Seed is always items[0] — input order is the caller's lever for
    deterministic ordering."""

    items = ["seed", "x", "y", "z"]
    embeds = [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ]
    assert select_diverse(items, embeds, k=1) == ["seed"]


def test_deterministic_for_same_input():
    items = list("abcdefgh")
    embeds = [[math.cos(i * 0.7), math.sin(i * 0.7)] for i in range(len(items))]
    a = select_diverse(items, embeds, k=4)
    b = select_diverse(items, embeds, k=4)
    assert a == b


def test_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k must be positive"):
        select_diverse(["a"], [[1.0]], k=0)


def test_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        select_diverse(["a", "b"], [[1.0]], k=1)


def test_empty_input_returns_empty():
    assert select_diverse([], [], k=3) == []


def test_handles_zero_norm_vectors():
    """Zero-norm embeddings shouldn't crash the cosine path — degenerate
    rows are treated as max-distance and surface during selection."""

    items = ["zero", "x", "y"]
    embeds = [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
    selected = select_diverse(items, embeds, k=2)
    # Seed is "zero"; the next pick should be either "x" or "y" —
    # both are at unit distance from the zero vector after the safe
    # normalisation. Either is acceptable.
    assert selected[0] == "zero"
    assert selected[1] in {"x", "y"}
