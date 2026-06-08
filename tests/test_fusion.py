"""Unit tests for the Python-side RRF fusion helper."""

from __future__ import annotations

from src.search.fusion import rrf_merge


def test_rrf_merge_single_path() -> None:
    rankings = {"dense": ["A", "B", "C"]}
    out = rrf_merge(rankings, rrf_k=60, top_k=2)
    assert out == ["A", "B"]


def test_rrf_merge_combines_paths() -> None:
    rankings = {
        "dense": ["A", "B", "C"],
        "sparse": ["B", "C", "D"],
    }
    out = rrf_merge(rankings, rrf_k=60, top_k=4)
    # B appears in both at ranks 2 and 1 -> highest fused RRF score.
    assert out[0] == "B"
    # C appears in both at ranks 3 and 2 -> second highest.
    assert out[1] == "C"
    # A and D each appear once; A at rank 1 should outrank D at rank 3.
    assert out[2] == "A"
    assert out[3] == "D"


def test_rrf_merge_excludes_dropped_path() -> None:
    """A path absent from the input is not considered, even if keys overlap."""
    out = rrf_merge({"dense": ["A"]}, rrf_k=60, top_k=2)
    assert out == ["A"]


def test_rrf_merge_top_k_truncates() -> None:
    rankings = {"dense": ["A", "B", "C", "D", "E"]}
    out = rrf_merge(rankings, rrf_k=60, top_k=3)
    assert out == ["A", "B", "C"]


def test_rrf_merge_empty_rankings() -> None:
    """No active paths -> empty output."""
    out = rrf_merge({}, rrf_k=60, top_k=5)
    assert out == []


def test_rrf_merge_deterministic_tie_break() -> None:
    """When two keys have equal fused scores, ties break by first-seen order."""
    # A at rank 1 in dense; B at rank 1 in sparse. Equal RRF scores.
    out = rrf_merge(
        {"dense": ["A"], "sparse": ["B"]},
        rrf_k=60,
        top_k=2,
    )
    assert out == ["A", "B"]
