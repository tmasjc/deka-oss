"""Shared fixtures for src.anchor tests.

Builds a synthetic Phase-1 session triple (progress log + details
sidecar + span cache) on disk so loader-side tests don't need a
live Milvus or BGE-M3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _config_block() -> dict[str, Any]:
    return {
        "rrf_k": 60,
        "per_path_limit": 20,
        "top_k": 10,
        "active_paths": ["dense", "sparse"],
    }


def _result(
    rank: int,
    pk: str,
    rating: str,
    *,
    sample_id: str = "S1",
    span_text: str = "",
    span_line_indices: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "pk": pk,
        "chunk_id": f"{sample_id}_C{rank:07d}",
        "chunk_content": f"content for {pk}",
        "sample_id": sample_id,
        "counselor_id": "C-1",
        "term": "2025F",
        "rating": rating,
        "source_paths": ["dense"],
        "scores": {"dense": 0.9, "sparse": 0.0},
        "span_text": span_text,
        "span_line_indices": span_line_indices or [],
    }


def _progress_turn(turn: int, fits: list[dict], not_fits: list[dict]) -> dict[str, Any]:
    rows = fits + not_fits
    return {
        "turn": turn,
        "timestamp": f"2026-04-23T10:0{turn}:00Z",
        "session_id": "SESS-TEST",
        "query": "test query",
        "config": _config_block(),
        "results": rows,
    }


def _details_turn(turn: int) -> dict[str, Any]:
    return {
        "turn": turn,
        "timestamp": f"2026-04-23T10:0{turn}:00Z",
        "session_id": "SESS-TEST",
        "query": "test query",
        "search": {
            "query": "test query",
            "collection": "parent_transcripts",
            "embed_url": "http://localhost:8000",
            "milvus_uri": "http://localhost:19530",
            "ranker_params": {"rrf_k": 60},
            "per_path_limit": 20,
            "top_k": 10,
            "active_paths": ["dense", "sparse"],
            "seen_set_size": 0,
            "milvus_search_params": {},
            "embedding": {
                "latency_ms": 1.0,
                "dense_dim": 4,
                "sparse_nnz": 0,
                "sparse_empty": True,
                "reused_from_probe": False,
            },
            "probes": {},
        },
    }


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    """Return a runs/ directory holding a complete fixture session."""
    runs = tmp_path / "runs"
    runs.mkdir()

    fit_rows = [
        _result(
            1,
            "pk-A",
            "FIT",
            span_text="span for A",
            span_line_indices=[0, 1],
        ),
        _result(
            2,
            "pk-B",
            "FIT",
            span_text="span for B",
            span_line_indices=[1],
        ),
    ]
    notfit_rows = [_result(3, "pk-C", "NOT_FIT")]

    progress = runs / "SESS-TEST.jsonl"
    with progress.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(_progress_turn(1, fit_rows, notfit_rows)) + "\n")
        fp.write(
            json.dumps(
                _progress_turn(
                    2,
                    [
                        _result(
                            1,
                            "pk-D",
                            "FIT",
                            span_text="span for D",
                            span_line_indices=[2, 3],
                        ),
                    ],
                    [],
                )
            )
            + "\n"
        )

    details = runs / "SESS-TEST.details.jsonl"
    with details.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(_details_turn(1)) + "\n")
        fp.write(json.dumps(_details_turn(2)) + "\n")

    span_cache = runs / "SESS-TEST.span_cache.jsonl"
    with span_cache.open("w", encoding="utf-8") as fp:
        # Match what's in the progress log so drift warnings stay quiet.
        for row in fit_rows:
            fp.write(
                json.dumps(
                    {
                        "chunk_id": row["chunk_id"],
                        "span_text": row["span_text"],
                        "span_line_indices": row["span_line_indices"],
                        "reason": "fixture",
                    }
                )
                + "\n"
            )
        fp.write(
            json.dumps(
                {
                    "chunk_id": "S1_C0000001",  # turn-2 row
                    "span_text": "span for D",
                    "span_line_indices": [2, 3],
                    "reason": "fixture",
                }
            )
            + "\n"
        )
    return runs
