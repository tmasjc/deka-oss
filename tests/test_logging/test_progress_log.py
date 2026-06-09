"""Tests for ``src.logging.progress_log.ProgressLogWriter``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.logging import ProgressLogWriter
from src.logging.run_logger import setup_run_logging
from src.search.config import SearchConfig
from src.search.evidence import CandidateRow, EvidenceRow, EvidenceTable
from src.session.state import SessionState, TurnRecord


def _config() -> SearchConfig:
    return SearchConfig(rrf_k=60, per_path_limit=20, top_k=3)


def _row(rank: int, rating: str = "FIT") -> EvidenceRow:
    return EvidenceRow(
        rank=rank,
        pk=100 + rank,
        chunk_id=f"abc_C000000{rank}",
        chunk_content=f"content {rank}",
        sample_id="abc",
        counselor_id="T0001",
        term="2025F",
        source_paths=["dense", "sparse"],
        scores={"dense": 0.8, "sparse": 0.7},
        rating=rating,
    )


def _table(search_diagnostics: dict | None = None) -> EvidenceTable:
    return EvidenceTable(
        query="test query",
        config=_config(),
        rows=[_row(1), _row(2, "NOT_FIT"), _row(3)],
        search_diagnostics=search_diagnostics,
    )


def _turn(
    reflection: dict | None,
    diag: dict | None = None,
    *,
    audit_turn: bool = False,
) -> TurnRecord:
    table = _table(search_diagnostics=diag)
    return TurnRecord(
        turn_number=1,
        query=table.query,
        config=table.config,
        evidence_table=table,
        breakdown={
            "dense_only": {"total": 0, "fit": 0, "not_fit": 0},
            "sparse_only": {"total": 0, "fit": 0, "not_fit": 0},
            "multi_path": {"total": 3, "fit": 2, "not_fit": 1},
        },
        precision=2 / 3,
        reflection=reflection,
        audit_turn=audit_turn,
    )


def _state() -> SessionState:
    return SessionState(current_config=_config(), session_id="sess_test1234")


def test_canonical_log_matches_schema(tmp_path: Path) -> None:
    writer = ProgressLogWriter("sess_test1234", tmp_path)
    state = _state()
    turn = _turn(
        reflection={
            "observe": "obs",
            "diagnose": "diag",
            "hypothesis": "if...then",
            "previous_hypothesis_verdict": None,
            "status": "CONTINUE",
            "_diagnostics": {"messages": ["<redacted>"]},
        }
    )
    writer.log_turn(state, turn)

    lines = writer.canonical_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["turn"] == 1
    assert entry["session_id"] == "sess_test1234"
    assert entry["query"] == "test query"
    assert entry["config"]["rrf_k"] == 60
    assert entry["config"]["active_paths"] == ["dense", "sparse"]
    assert "weights" not in entry["config"]
    assert "ranker" not in entry["config"]
    # audit_turn is recorded for every entry; default False for regular turns.
    assert entry["audit_turn"] is False
    assert entry["metrics"]["total"] == 3
    assert entry["metrics"]["fit_count"] == 2
    assert entry["metrics"]["precision_at_k"] == pytest.approx(2 / 3)
    # Absent diagnostics defaults seen_set_size to 0.
    assert entry["metrics"]["seen_set_size"] == 0
    assert len(entry["results"]) == 3
    # Canonical log strips the _diagnostics key.
    assert "_diagnostics" not in entry["reflection"]
    # Reflection is now a reasoning trace — no prescribe block.
    assert "prescribe" not in entry["reflection"]
    assert entry["reflection"]["hypothesis"] == "if...then"


def test_canonical_log_records_audit_turn_flag(tmp_path: Path) -> None:
    writer = ProgressLogWriter("sess_audit", tmp_path)
    state = _state()
    turn = _turn(reflection=None, audit_turn=True)
    writer.log_turn(state, turn)

    entry = json.loads(writer.canonical_path.read_text().splitlines()[0])
    assert entry["audit_turn"] is True


def test_details_log_carries_diagnostics(tmp_path: Path) -> None:
    writer = ProgressLogWriter("sess_test1234", tmp_path)
    state = _state()
    search_diag = {
        "embedding": {"latency_ms": 12.3, "dense_dim": 1024, "sparse_nnz": 42},
        "probes": {
            "dense": {"hit_count": 20, "score_max": 0.9},
            "sparse": {"hit_count": 10, "score_max": 0.6},
        },
    }
    refl_diag = {
        "model": "openai/gpt-4o",
        "temperature": 0.2,
        "messages": [{"role": "system", "content": "You are..."}],
        "raw_response": '{"observe": "..."}',
    }
    turn = _turn(
        reflection={"observe": "o", "diagnose": "d", "_diagnostics": refl_diag},
        diag=search_diag,
    )
    writer.log_turn(state, turn)

    entry = json.loads(writer.details_path.read_text(encoding="utf-8").splitlines()[0])
    assert entry["search"]["embedding"]["dense_dim"] == 1024
    assert entry["search"]["probes"]["dense"]["hit_count"] == 20
    assert entry["reflection"]["model"] == "openai/gpt-4o"
    assert entry["reflection"]["messages"][0]["role"] == "system"
    assert entry["reflection"]["raw_response"].startswith('{"observe"')
    # config_diff used to be sourced from the prescription validator's
    # diagnostics; with reflection no longer prescribing, it stays None.
    assert entry["config_diff"] is None


def test_canonical_log_captures_seen_set_size(tmp_path: Path) -> None:
    """When search diagnostics carry ``seen_set_size``, it reaches the
    canonical log's metrics block for post-hoc analysis of dedup impact."""
    writer = ProgressLogWriter("sess_test1234", tmp_path)
    state = _state()
    turn = _turn(reflection=None, diag={"seen_set_size": 47})
    writer.log_turn(state, turn)

    entry = json.loads(writer.canonical_path.read_text().splitlines()[0])
    assert entry["metrics"]["seen_set_size"] == 47


def test_handles_missing_reflection(tmp_path: Path) -> None:
    writer = ProgressLogWriter("sess_test1234", tmp_path)
    state = _state()
    turn = _turn(reflection=None)
    writer.log_turn(state, turn)

    canonical = json.loads(writer.canonical_path.read_text().splitlines()[0])
    assert canonical["reflection"] is None

    details = json.loads(writer.details_path.read_text().splitlines()[0])
    assert details["reflection"] is None
    assert details["search"] is None


def test_append_only_across_turns(tmp_path: Path) -> None:
    writer = ProgressLogWriter("sess_test1234", tmp_path)
    state = _state()

    turn1 = _turn(reflection=None)
    writer.log_turn(state, turn1)
    turn2 = _turn(reflection={"observe": "second"})
    turn2.turn_number = 2
    writer.log_turn(state, turn2)

    lines = writer.canonical_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["turn"] == 1
    assert json.loads(lines[1])["turn"] == 2
    # First line must not have been rewritten
    assert json.loads(lines[0])["reflection"] is None


def test_span_fields_round_trip_through_canonical_log(tmp_path: Path) -> None:
    """Fused rows and per-path candidates persist span_line_indices + span_text."""
    writer = ProgressLogWriter("sess_spans", tmp_path)
    state = _state()

    row = _row(1, "FIT")
    row.span_line_indices = [1, 2]
    row.span_text = "老师：concept line\n家长：reply"

    candidate = CandidateRow(
        path="dense",
        rank_in_path=1,
        pk=999,
        chunk_id="xyz_C0000001",
        chunk_content="老师：ask\n家长：answer",
        sample_id="xyz",
        counselor_id="T0002",
        term="2025F",
        score=0.75,
        rating="NOT_FIT",
        span_line_indices=[0],
        span_text="老师：ask",
    )
    table = EvidenceTable(
        query="q",
        config=_config(),
        rows=[row],
        per_path_candidates={"dense": [candidate], "sparse": []},
    )
    turn = TurnRecord(
        turn_number=1,
        query=table.query,
        config=table.config,
        evidence_table=table,
        breakdown={
            "dense_only": {"total": 1, "fit": 1, "not_fit": 0},
            "sparse_only": {"total": 0, "fit": 0, "not_fit": 0},
            "multi_path": {"total": 0, "fit": 0, "not_fit": 0},
        },
        precision=1.0,
        reflection=None,
    )
    writer.log_turn(state, turn)

    entry = json.loads(
        writer.canonical_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert entry["results"][0]["span_line_indices"] == [1, 2]
    assert entry["results"][0]["span_text"] == "老师：concept line\n家长：reply"
    dense_cand = entry["per_path_candidates"]["dense"][0]
    assert dense_cand["span_line_indices"] == [0]
    assert dense_cand["span_text"] == "老师：ask"


def test_span_extraction_block_written_to_details(tmp_path: Path) -> None:
    """When run_search set search_diagnostics['span_extraction'], the details
    sidecar entry carries it verbatim so downstream analysis has the
    model/prompt_version/latency/counts block."""
    writer = ProgressLogWriter("sess_span_diag", tmp_path)
    state = _state()

    block = {
        "model": "openai/gpt-5.4-mini",
        "prompt_version": "v1",
        "latency_ms_total": 123.4,
        "chunks_extracted": 10,
        "cache_hits": 6,
        "empty_span_count": 2,
        "span_length_distribution": {"0": 2, "1": 5, "2": 3, "3": 0},
    }
    turn = _turn(reflection=None, diag={"span_extraction": block})
    writer.log_turn(state, turn)

    entry = json.loads(writer.details_path.read_text(encoding="utf-8").splitlines()[0])
    assert entry["search"]["span_extraction"] == block


def test_span_fields_default_empty_when_unset(tmp_path: Path) -> None:
    """Rows constructed without span fields emit [] and "" in the log."""
    writer = ProgressLogWriter("sess_spans_empty", tmp_path)
    state = _state()
    turn = _turn(reflection=None)
    writer.log_turn(state, turn)

    entry = json.loads(
        writer.canonical_path.read_text(encoding="utf-8").splitlines()[0]
    )
    for result in entry["results"]:
        assert result["span_line_indices"] == []
        assert result["span_text"] == ""


def test_log_event_appends_to_details_only(tmp_path: Path) -> None:
    """``log_event`` lands a free-form entry in the sidecar without
    touching the canonical jsonl. Used for post-hoc operator decisions
    (e.g. agent-recommendation apply/ignore)."""
    writer = ProgressLogWriter("sess_event_test", tmp_path)

    writer.log_event(
        turn=2,
        kind="path_drop_recommendation_decision",
        recommendation={"path": "dense", "reason": "r", "confidence": "medium"},
        decision="ignore",
    )

    # Canonical jsonl is untouched (file may not even exist).
    assert not writer.canonical_path.exists()

    lines = writer.details_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["turn"] == 2
    assert entry["kind"] == "path_drop_recommendation_decision"
    assert entry["session_id"] == "sess_event_test"
    assert entry["decision"] == "ignore"
    assert entry["recommendation"]["path"] == "dense"
    assert "timestamp" in entry


def test_log_event_does_not_disturb_canonical_when_log_turn_also_runs(
    tmp_path: Path,
) -> None:
    """A subsequent ``log_turn`` writes to the canonical jsonl unchanged
    even after a sidecar event has been emitted on the same session."""
    writer = ProgressLogWriter("sess_mixed", tmp_path)

    writer.log_event(
        turn=1, kind="path_drop_recommendation_decision", decision="ignore"
    )
    writer.log_turn(
        _state(),
        _turn(
            reflection={
                "observe": "o",
                "diagnose": "d",
                "hypothesis": "h",
                "previous_hypothesis_verdict": None,
                "status": "CONTINUE",
            }
        ),
    )

    canonical_lines = writer.canonical_path.read_text(encoding="utf-8").splitlines()
    details_lines = writer.details_path.read_text(encoding="utf-8").splitlines()
    # Canonical has exactly one turn record; details has the event +
    # the per-turn details record (always written by log_turn).
    assert len(canonical_lines) == 1
    assert len(details_lines) == 2
    assert json.loads(details_lines[0])["kind"] == "path_drop_recommendation_decision"


def test_log_converged_writes_event_row(tmp_path: Path) -> None:
    """``log_converged`` lands a single ``{"event":"converged",...}``
    row on the canonical jsonl. The resume classifier reads only the
    file's last line, so the row format is the contract."""
    writer = ProgressLogWriter("sess_marker", tmp_path)
    writer.log_converged(turn=3)

    lines = writer.canonical_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry == {"event": "converged", "turn": 3, "ts": entry["ts"]}
    assert "ts" in entry and entry["ts"].endswith("Z")


def test_log_turn_auto_emits_marker_when_state_is_converged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``log_turn`` calls ``log_converged`` automatically whenever
    ``state.is_converged`` is True after the turn row is written.

    The fixture turn carries precision=2/3 with 2 unique FIT pks; we
    monkeypatch the thresholds so this fixture trips the dual gate."""
    monkeypatch.setattr("src.session.state.CONVERGENCE_PRECISION", 0.5)
    monkeypatch.setattr("src.session.state.CONVERGENCE_MIN_FIT", 1)
    monkeypatch.setattr("src.session.state.CONVERGENCE_MIN_NOT_FIT", 0)

    writer = ProgressLogWriter("sess_converged", tmp_path)
    state = _state()
    turn = _turn(reflection=None)
    # Simulate "this turn just completed" — the writer reads the turn
    # via state.turns to compute is_converged.
    state.turns.append(turn)
    assert state.is_converged

    writer.log_turn(state, turn)

    lines = writer.canonical_path.read_text(encoding="utf-8").splitlines()
    # Turn row followed by the converged marker.
    assert len(lines) == 2
    assert json.loads(lines[0])["turn"] == turn.turn_number
    last = json.loads(lines[-1])
    assert last["event"] == "converged"
    assert last["turn"] == turn.turn_number


def test_log_turn_does_not_emit_marker_when_not_converged(
    tmp_path: Path,
) -> None:
    """A regular sub-convergence turn writes only the canonical turn row;
    no event marker is appended."""
    writer = ProgressLogWriter("sess_below", tmp_path)
    state = _state()
    turn = _turn(reflection=None)
    state.turns.append(turn)
    # Default thresholds (loaded from config.yaml) are stricter than this
    # fixture meets, so is_converged stays False without monkeypatching.
    assert not state.is_converged

    writer.log_turn(state, turn)

    lines = writer.canonical_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "event" not in json.loads(lines[0])


def test_log_turn_re_emits_marker_after_each_post_convergence_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The marker re-emits after every turn while the state stays
    converged, so the canonical jsonl's last line is always the
    marker — required for O(read-last-line) classification when the
    operator opts not to harvest and runs another turn."""
    monkeypatch.setattr("src.session.state.CONVERGENCE_PRECISION", 0.5)
    monkeypatch.setattr("src.session.state.CONVERGENCE_MIN_FIT", 1)
    monkeypatch.setattr("src.session.state.CONVERGENCE_MIN_NOT_FIT", 0)

    writer = ProgressLogWriter("sess_post_conv", tmp_path)
    state = _state()

    turn1 = _turn(reflection=None)
    turn1.turn_number = 1
    state.turns.append(turn1)
    writer.log_turn(state, turn1)

    turn2 = _turn(reflection=None)
    turn2.turn_number = 2
    state.turns.append(turn2)
    writer.log_turn(state, turn2)

    lines = writer.canonical_path.read_text(encoding="utf-8").splitlines()
    # Layout: turn1, marker(turn=1), turn2, marker(turn=2).
    assert len(lines) == 4
    assert "event" not in json.loads(lines[0])
    assert json.loads(lines[1]) == {
        "event": "converged",
        "turn": 1,
        "ts": json.loads(lines[1])["ts"],
    }
    assert "event" not in json.loads(lines[2])
    last = json.loads(lines[-1])
    assert last["event"] == "converged"
    assert last["turn"] == 2


def test_setup_run_logging_writes_to_file(tmp_path: Path) -> None:
    import logging

    path = setup_run_logging("sess_run_test", tmp_path)
    assert path == tmp_path / "sess_run_test.log"

    logging.getLogger("test.logger").warning("hello world")

    # Force flush of all handlers
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = path.read_text()
    assert "hello world" in content
    assert "WARNING" in content
