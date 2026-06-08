"""Tests for ``src.replay.loader``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.replay.loader import ReplayLoadError, load_session
from src.search.config import SearchConfig
from src.search.evidence import EvidenceRow, EvidenceTable
from src.logging import ProgressLogWriter
from src.session.state import SessionState, TurnRecord


def _config(per_path_limit: int = 20) -> SearchConfig:
    return SearchConfig(rrf_k=60, per_path_limit=per_path_limit, top_k=2)


def _row(rank: int, rating: str = "FIT") -> EvidenceRow:
    return EvidenceRow(
        rank=rank,
        pk=f"pk-{rank}",
        chunk_id=f"sample_C000000{rank}",
        chunk_content=f"老师：line for rank {rank}\n家长：reply {rank}",
        sample_id="sample",
        counselor_id="T0001",
        term="2025F",
        source_paths=["dense"],
        scores={"dense": 0.8, "sparse": 0.0},
        rating=rating,
    )


def _table(
    config: SearchConfig, search_diagnostics: dict | None = None
) -> EvidenceTable:
    return EvidenceTable(
        query="家长觉得课程有点难",
        config=config,
        rows=[_row(1, "FIT"), _row(2, "NOT_FIT")],
        search_diagnostics=search_diagnostics,
    )


def _turn(turn_no: int, config: SearchConfig, reflection: dict | None) -> TurnRecord:
    diag = {
        "embedding": {"latency_ms": 12.0, "dense_dim": 1024, "sparse_nnz": 4},
        "probes": {
            "dense": {
                "skipped": False,
                "hit_count": 20,
                "latency_ms": 100.0,
                "score_min": 0.7,
                "score_max": 0.9,
                "score_mean": 0.8,
                "top3_pks": ["pk-1"],
                "top3_entities": [],
                "filtered_by_seen": 0,
            },
            "sparse": {
                "skipped": False,
                "hit_count": 5,
                "latency_ms": 50.0,
                "score_min": 0.2,
                "score_max": 0.4,
                "score_mean": 0.3,
                "top3_pks": [],
                "top3_entities": [],
                "filtered_by_seen": 0,
            },
        },
        "seen_set_size": 0,
    }
    table = _table(config, search_diagnostics=diag)
    return TurnRecord(
        turn_number=turn_no,
        query=table.query,
        config=config,
        evidence_table=table,
        breakdown={
            "dense_only": {"total": 2, "fit": 1, "not_fit": 1},
            "sparse_only": {"total": 0, "fit": 0, "not_fit": 0},
            "multi_path": {"total": 0, "fit": 0, "not_fit": 0},
        },
        precision=0.5,
        reflection=reflection,
    )


def _write_session(tmp_path: Path) -> str:
    """Drive ProgressLogWriter to produce a real two-turn session."""
    session_id = "sess_replay_test"
    writer = ProgressLogWriter(session_id, tmp_path)
    state = SessionState(current_config=_config(), session_id=session_id)

    for turn_no, per_path_limit in ((1, 20), (2, 40)):
        config = _config(per_path_limit=per_path_limit)
        reflection = {
            "observe": f"obs turn {turn_no}",
            "diagnose": f"diag turn {turn_no}",
            "hypothesis": "if X then Y because Z",
            "previous_hypothesis_verdict": "CONFIRMED" if turn_no == 2 else None,
            "status": "CONTINUE",
            "prescribe": {
                "rrf_k": 60,
                "per_path_limit": per_path_limit + 20,
                "top_k": 2,
                "active_paths": ["dense", "sparse"],
            },
            "_diagnostics": {
                "model": "openai/gpt-test",
                "raw_response": f"raw response turn {turn_no}",
                "validation": {"axes_changed": 1},
            },
        }
        writer.log_turn(state, _turn(turn_no, config, reflection))

    return session_id


def test_load_session_by_id(tmp_path: Path) -> None:
    session_id = _write_session(tmp_path)
    session = load_session(session_id, runs_dir=tmp_path)

    assert session.session_id == session_id
    assert len(session.turns) == 2

    turn1, turn2 = session.turns
    assert turn1.turn_number == 1
    assert turn1.query == "家长觉得课程有点难"
    assert turn1.config.per_path_limit == 20
    assert turn1.config.rrf_k == 60
    assert turn1.config.active_paths == frozenset({"dense", "sparse"})
    assert turn1.precision == 0.5

    rows = turn1.evidence_table.rows
    assert len(rows) == 2
    assert rows[0].rating == "FIT"
    assert rows[1].rating == "NOT_FIT"
    assert "老师" in rows[0].chunk_content
    assert rows[0].scores["dense"] == pytest.approx(0.8)

    assert turn1.reflection is not None
    assert turn1.reflection["hypothesis"] == "if X then Y because Z"
    # Diagnostics from the sidecar are merged in for the modal.
    assert turn1.reflection["_diagnostics"]["raw_response"] == "raw response turn 1"

    assert turn1.probe is not None
    assert turn1.probe.query == "家长觉得课程有点难"
    assert turn1.probe.stats_by_path["dense"]["hit_count"] == 20

    assert turn2.config.per_path_limit == 40
    assert turn2.reflection["previous_hypothesis_verdict"] == "CONFIRMED"


def test_load_session_by_canonical_path(tmp_path: Path) -> None:
    session_id = _write_session(tmp_path)
    session = load_session(tmp_path / f"{session_id}.jsonl")
    assert session.session_id == session_id
    assert len(session.turns) == 2


def test_load_session_missing_files_raises(tmp_path: Path) -> None:
    with pytest.raises(ReplayLoadError, match="Canonical log not found"):
        load_session("nope", runs_dir=tmp_path)

    (tmp_path / "halfsess.jsonl").write_text(
        json.dumps({"turn": 1, "session_id": "halfsess"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ReplayLoadError, match="Details sidecar not found"):
        load_session("halfsess", runs_dir=tmp_path)


def test_span_fields_round_trip_through_replay(tmp_path: Path) -> None:
    """Span fields written by the writer are reconstructed by the loader."""
    session_id = "sess_span_roundtrip"
    writer = ProgressLogWriter(session_id, tmp_path)
    state = SessionState(current_config=_config(), session_id=session_id)

    config = _config()
    row = _row(1, "FIT")
    row.span_line_indices = [0, 1]
    row.span_text = "老师：line for rank 1\n家长：reply 1"
    from src.search.evidence import CandidateRow

    candidate = CandidateRow(
        path="sparse",
        rank_in_path=1,
        pk="cand-pk-1",
        chunk_id="sample_C0000099",
        chunk_content="老师：ask\n家长：answer\n老师：follow-up",
        sample_id="sample",
        counselor_id="T0001",
        term="2025F",
        score=0.42,
        rating="FIT",
        span_line_indices=[2],
        span_text="老师：follow-up",
    )
    table = EvidenceTable(
        query="家长觉得课程有点难",
        config=config,
        rows=[row, _row(2, "NOT_FIT")],
        per_path_candidates={"dense": [], "sparse": [candidate]},
        search_diagnostics={
            "embedding": {"latency_ms": 10.0, "dense_dim": 1024, "sparse_nnz": 4},
            "probes": {
                "dense": {
                    "skipped": False,
                    "hit_count": 20,
                    "latency_ms": 1.0,
                    "score_min": 0.1,
                    "score_max": 0.9,
                    "score_mean": 0.5,
                    "top3_pks": [],
                    "top3_entities": [],
                    "filtered_by_seen": 0,
                },
                "sparse": {
                    "skipped": False,
                    "hit_count": 5,
                    "latency_ms": 1.0,
                    "score_min": 0.1,
                    "score_max": 0.5,
                    "score_mean": 0.3,
                    "top3_pks": [],
                    "top3_entities": [],
                    "filtered_by_seen": 0,
                },
            },
            "seen_set_size": 0,
        },
    )
    turn = TurnRecord(
        turn_number=1,
        query=table.query,
        config=config,
        evidence_table=table,
        breakdown={
            "dense_only": {"total": 1, "fit": 1, "not_fit": 0},
            "sparse_only": {"total": 1, "fit": 0, "not_fit": 1},
            "multi_path": {"total": 0, "fit": 0, "not_fit": 0},
        },
        precision=0.5,
        reflection=None,
    )
    writer.log_turn(state, turn)

    session = load_session(session_id, runs_dir=tmp_path)
    loaded_row = session.turns[0].evidence_table.rows[0]
    assert loaded_row.span_line_indices == [0, 1]
    assert loaded_row.span_text == "老师：line for rank 1\n家长：reply 1"

    loaded_cand = session.turns[0].evidence_table.per_path_candidates["sparse"][0]
    assert loaded_cand.span_line_indices == [2]
    assert loaded_cand.span_text == "老师：follow-up"


def test_legacy_logs_without_span_fields_load_with_defaults(tmp_path: Path) -> None:
    """Pre-span logs (fields absent) load with span_line_indices=[] and span_text=""."""
    canonical = tmp_path / "legacy.jsonl"
    details = tmp_path / "legacy.details.jsonl"
    entry = {
        "turn": 1,
        "session_id": "legacy",
        "query": "q",
        "config": {
            "rrf_k": 60,
            "per_path_limit": 20,
            "top_k": 1,
            "active_paths": ["dense"],
        },
        "results": [
            {
                "rank": 1,
                "pk": "pk-1",
                "chunk_id": "sample_C0000001",
                "chunk_content": "老师：hi\n家长：hello",
                "sample_id": "sample",
                "counselor_id": "T0001",
                "term": "2025F",
                "rating": "FIT",
                "source_paths": ["dense"],
                "scores": {"dense": 0.8, "sparse": 0.0},
            }
        ],
        "per_path_candidates": {
            "dense": [],
            "sparse": [
                {
                    "rank_in_path": 1,
                    "pk": "c-1",
                    "chunk_id": "sample_C0000002",
                    "chunk_content": "老师：a\n家长：b",
                    "sample_id": "sample",
                    "counselor_id": "T0001",
                    "term": "2025F",
                    "score": 0.5,
                    "rating": "NOT_FIT",
                }
            ],
        },
        "metrics": {
            "total": 1,
            "fit_count": 1,
            "not_fit_count": 0,
            "precision_at_k": 1.0,
            "per_path_breakdown": {},
            "seen_set_size": 0,
        },
        "reflection": None,
    }
    canonical.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    details.write_text(json.dumps({"turn": 1}) + "\n", encoding="utf-8")

    session = load_session("legacy", runs_dir=tmp_path)
    loaded_row = session.turns[0].evidence_table.rows[0]
    assert loaded_row.span_line_indices == []
    assert loaded_row.span_text == ""
    loaded_cand = session.turns[0].evidence_table.per_path_candidates["sparse"][0]
    assert loaded_cand.span_line_indices == []
    assert loaded_cand.span_text == ""


def test_load_session_rejects_pre_replay_logs(tmp_path: Path) -> None:
    """Logs written before chunk_content was persisted cannot be replayed."""
    canonical = tmp_path / "old.jsonl"
    details = tmp_path / "old.details.jsonl"
    canonical.write_text(
        json.dumps(
            {
                "turn": 1,
                "session_id": "old",
                "query": "q",
                "config": {
                    "rrf_k": 60,
                    "per_path_limit": 20,
                    "top_k": 1,
                    "active_paths": ["dense"],
                },
                "results": [
                    {
                        "rank": 1,
                        "pk": "pk-1",
                        "chunk_id": "sample_C0000001",
                        "sample_id": "sample",
                        "counselor_id": "T0001",
                        "term": "2025F",
                        "rating": "FIT",
                        "source_paths": ["dense"],
                        "scores": {"dense": 0.8, "sparse": 0.0},
                    }
                ],
                "per_path_candidates": {"dense": [], "sparse": []},
                "metrics": {
                    "total": 1,
                    "fit_count": 1,
                    "not_fit_count": 0,
                    "precision_at_k": 1.0,
                    "per_path_breakdown": {},
                    "seen_set_size": 0,
                },
                "reflection": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    details.write_text("", encoding="utf-8")

    with pytest.raises(ReplayLoadError, match="predates the replay-era writer fix"):
        load_session("old", runs_dir=tmp_path)


# ---------------------------------------------------------------------------
# Hard cutover: BM25 was removed from Phase 1 after commit 7237be0. Logs
# that still carry "bm25" in active_paths, scores, or per_path_candidates
# can no longer be replayed by current builds — the loader raises early
# with a turn-numbered error so users know which build to check out.
# ---------------------------------------------------------------------------


def _legacy_canonical_entry(**overrides) -> dict:
    base = {
        "turn": 1,
        "session_id": "legacy_bm25",
        "query": "q",
        "config": {
            "rrf_k": 60,
            "per_path_limit": 20,
            "top_k": 1,
            "active_paths": ["dense", "sparse"],
        },
        "results": [
            {
                "rank": 1,
                "pk": "pk-1",
                "chunk_id": "sample_C0000001",
                "chunk_content": "老师：hi\n家长：hello",
                "sample_id": "sample",
                "counselor_id": "T0001",
                "term": "2025F",
                "rating": "FIT",
                "source_paths": ["dense"],
                "scores": {"dense": 0.8, "sparse": 0.0},
            }
        ],
        "per_path_candidates": {"dense": [], "sparse": []},
        "metrics": {
            "total": 1,
            "fit_count": 1,
            "not_fit_count": 0,
            "precision_at_k": 1.0,
            "per_path_breakdown": {},
            "seen_set_size": 0,
        },
        "reflection": None,
    }
    # Shallow-merge overrides into the deep config / results structure.
    if "config" in overrides:
        base["config"] = {**base["config"], **overrides.pop("config")}
    if "results_scores" in overrides:
        base["results"][0]["scores"] = overrides.pop("results_scores")
    if "per_path_candidates" in overrides:
        base["per_path_candidates"] = overrides.pop("per_path_candidates")
    base.update(overrides)
    return base


def _write_legacy_session(tmp_path: Path, entry: dict) -> str:
    canonical = tmp_path / "legacy_bm25.jsonl"
    details = tmp_path / "legacy_bm25.details.jsonl"
    canonical.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    details.write_text(json.dumps({"turn": 1}) + "\n", encoding="utf-8")
    return "legacy_bm25"


def test_legacy_bm25_active_paths_rejected(tmp_path: Path) -> None:
    entry = _legacy_canonical_entry(
        config={"active_paths": ["bm25", "dense", "sparse"]},
    )
    sid = _write_legacy_session(tmp_path, entry)
    with pytest.raises(ReplayLoadError, match=r"Turn 1: legacy 'bm25' in 'active_paths'"):
        load_session(sid, runs_dir=tmp_path)


def test_legacy_bm25_in_scores_rejected(tmp_path: Path) -> None:
    entry = _legacy_canonical_entry(
        results_scores={"dense": 0.8, "sparse": 0.0, "bm25": 0.0},
    )
    sid = _write_legacy_session(tmp_path, entry)
    with pytest.raises(ReplayLoadError, match=r"Turn 1: legacy 'bm25' in 'scores.bm25'"):
        load_session(sid, runs_dir=tmp_path)


def test_legacy_bm25_in_candidates_rejected(tmp_path: Path) -> None:
    entry = _legacy_canonical_entry(
        per_path_candidates={"dense": [], "sparse": [], "bm25": []},
    )
    sid = _write_legacy_session(tmp_path, entry)
    with pytest.raises(
        ReplayLoadError,
        match=r"Turn 1: legacy 'bm25' in 'per_path_candidates.bm25'",
    ):
        load_session(sid, runs_dir=tmp_path)


def test_legacy_bm25_error_names_pre_removal_sha(tmp_path: Path) -> None:
    """The error message must point operators at the cutover commit so they
    can git-checkout a build that can replay the old log."""
    entry = _legacy_canonical_entry(
        config={"active_paths": ["bm25", "dense", "sparse"]},
    )
    sid = _write_legacy_session(tmp_path, entry)
    with pytest.raises(ReplayLoadError, match="7237be0"):
        load_session(sid, runs_dir=tmp_path)
