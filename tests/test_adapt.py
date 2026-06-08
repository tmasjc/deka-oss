"""Unit tests for the diagnostics-only adapt step."""

from __future__ import annotations

from typing import Any

import pytest

from src.search import (
    AdaptError,
    ProbeResult,
    SearchConfig,
    adapt_config,
)


def _make_config(**overrides: Any) -> SearchConfig:
    base: dict[str, Any] = {
        "rrf_k": 60,
        "per_path_limit": 20,
        "top_k": 10,
    }
    base.update(overrides)
    return SearchConfig(**base)


def _stats(
    hit_count: int = 20,
    score_min: float = 0.7,
    score_max: float = 0.9,
    score_mean: float = 0.8,
    skipped: bool = False,
) -> dict[str, Any]:
    return {
        "skipped": skipped,
        "hit_count": hit_count,
        "score_min": score_min,
        "score_max": score_max,
        "score_mean": score_mean,
        "top3_pks": [],
        "latency_ms": 1.0,
    }


def _probe(stats_by_path: dict[str, dict[str, Any]]) -> ProbeResult:
    return ProbeResult(
        query="q",
        config=_make_config(),
        dense_vec=[0.0],
        sparse_vec={},
        provenance={},
        scores_by_path={"dense": {}, "sparse": {}},
        stats_by_path=stats_by_path,
        embed_diagnostics={},
    )


class TestAdaptConfig:
    def test_all_paths_active_no_flags(self) -> None:
        seed = _make_config()
        probe = _probe(
            {
                "dense": _stats(hit_count=20, score_mean=0.8),
                "sparse": _stats(hit_count=20, score_mean=0.5),
            }
        )

        adapted = adapt_config(seed, probe)

        # Diagnostics-only: config is unchanged.
        assert adapted.config == seed
        assert adapted.rationale == []
        assert adapted.flags == []

    def test_dead_path_emits_rationale(self) -> None:
        seed = _make_config()
        probe = _probe(
            {
                "dense": _stats(hit_count=20),
                "sparse": _stats(hit_count=0),
            }
        )

        adapted = adapt_config(seed, probe)

        assert adapted.config == seed
        assert any("sparse" in r for r in adapted.rationale)
        # One active path → single-path flag fires.
        assert any("single-path start" in flag for flag in adapted.flags)

    def test_skipped_path_treated_as_dead(self) -> None:
        seed = _make_config()
        probe = _probe(
            {
                "dense": _stats(hit_count=20),
                "sparse": _stats(skipped=True, hit_count=0),
            }
        )

        adapted = adapt_config(seed, probe)

        assert any("sparse" in r for r in adapted.rationale)

    def test_all_paths_dead_raises(self) -> None:
        seed = _make_config()
        probe = _probe(
            {
                "dense": _stats(hit_count=0),
                "sparse": _stats(skipped=True, hit_count=0),
            }
        )

        with pytest.raises(AdaptError, match="Both retrieval paths"):
            adapt_config(seed, probe)

    def test_per_path_limit_and_top_k_pass_through(self) -> None:
        seed = _make_config(per_path_limit=42, top_k=7, rrf_k=120)
        probe = _probe(
            {
                "dense": _stats(hit_count=20),
                "sparse": _stats(hit_count=0),
            }
        )

        adapted = adapt_config(seed, probe)

        # adapt_config is diagnostics-only — config fields must be identical.
        assert adapted.config.per_path_limit == 42
        assert adapted.config.top_k == 7
        assert adapted.config.rrf_k == 120
