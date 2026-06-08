"""Tests for src.replay.metrics.is_session_converged (triple gate)."""

from __future__ import annotations

from pathlib import Path

from src.replay.loader import ReplaySession, ReplayTurn
from src.replay.metrics import (
    CONVERGENCE_MIN_FIT,
    CONVERGENCE_MIN_NOT_FIT,
    CONVERGENCE_PRECISION,
    is_session_converged,
)
from src.search.config import SearchConfig
from src.search.evidence import EvidenceRow, EvidenceTable


def _rows(n_fit: int, n_not_fit: int, pk_start: int = 1) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    pk = pk_start
    for _ in range(n_fit):
        rows.append(
            EvidenceRow(
                rank=pk,
                pk=pk,
                chunk_id=f"C{pk:07d}",
                chunk_content="...",
                sample_id="S_x",
                counselor_id="C-1",
                term="t",
                source_paths=["dense"],
                scores={"dense": 0.9, "sparse": 0.0, "bm25": 0.0},
                rating="FIT",
            )
        )
        pk += 1
    for _ in range(n_not_fit):
        rows.append(
            EvidenceRow(
                rank=pk,
                pk=pk,
                chunk_id=f"C{pk:07d}",
                chunk_content="...",
                sample_id="S_x",
                counselor_id="C-1",
                term="t",
                source_paths=["dense"],
                scores={"dense": 0.1, "sparse": 0.0, "bm25": 0.0},
                rating="NOT_FIT",
            )
        )
        pk += 1
    return rows


def _turn(
    precision: float, turn_number: int, *, rows: list[EvidenceRow] | None = None
) -> ReplayTurn:
    config = SearchConfig(
        rrf_k=60,
        per_path_limit=10,
        top_k=10,
        active_paths=frozenset({"dense", "sparse", "bm25"}),
    )
    table = EvidenceTable(query="q", config=config, rows=rows or [])
    return ReplayTurn(
        turn_number=turn_number,
        timestamp="",
        query="q",
        config=config,
        evidence_table=table,
        breakdown={},
        precision=precision,
        reflection=None,
        probe=None,
    )


def _session(turns: list[ReplayTurn]) -> ReplaySession:
    return ReplaySession(
        session_id="x",
        canonical_path=Path("x.jsonl"),
        details_path=Path("x.details.jsonl"),
        turns=turns,
    )


def test_empty_session_is_not_converged():
    assert is_session_converged(_session([])) is False


def test_single_turn_meeting_all_gates_is_converged():
    rows = _rows(n_fit=CONVERGENCE_MIN_FIT, n_not_fit=CONVERGENCE_MIN_NOT_FIT)
    session = _session(
        [_turn(precision=CONVERGENCE_PRECISION, turn_number=1, rows=rows)]
    )
    assert is_session_converged(session) is True


def test_precision_gate_unmet_not_converged():
    rows = _rows(n_fit=CONVERGENCE_MIN_FIT + 5, n_not_fit=CONVERGENCE_MIN_NOT_FIT)
    session = _session(
        [_turn(precision=CONVERGENCE_PRECISION - 0.1, turn_number=1, rows=rows)]
    )
    assert is_session_converged(session) is False


def test_fit_gate_unmet_not_converged():
    rows = _rows(n_fit=CONVERGENCE_MIN_FIT - 1, n_not_fit=CONVERGENCE_MIN_NOT_FIT)
    session = _session(
        [_turn(precision=CONVERGENCE_PRECISION, turn_number=1, rows=rows)]
    )
    assert is_session_converged(session) is False


def test_not_fit_gate_unmet_not_converged():
    """P@K + FIT satisfied but zero NOT_FITs → not converged."""
    rows = _rows(n_fit=CONVERGENCE_MIN_FIT, n_not_fit=0)
    session = _session(
        [_turn(precision=CONVERGENCE_PRECISION, turn_number=1, rows=rows)]
    )
    assert is_session_converged(session) is False


def test_not_fit_gate_just_below_floor_not_converged():
    """One NOT_FIT short of the floor still fails the gate."""
    rows = _rows(
        n_fit=CONVERGENCE_MIN_FIT, n_not_fit=CONVERGENCE_MIN_NOT_FIT - 1
    )
    session = _session(
        [_turn(precision=CONVERGENCE_PRECISION, turn_number=1, rows=rows)]
    )
    assert is_session_converged(session) is False


def test_cumulative_fit_across_turns():
    rows_t1 = _rows(n_fit=8, n_not_fit=CONVERGENCE_MIN_NOT_FIT, pk_start=1)
    rows_t2 = _rows(n_fit=CONVERGENCE_MIN_FIT - 8, n_not_fit=0, pk_start=100)
    session = _session(
        [
            _turn(precision=CONVERGENCE_PRECISION, turn_number=1, rows=rows_t1),
            _turn(precision=CONVERGENCE_PRECISION, turn_number=2, rows=rows_t2),
        ]
    )
    assert is_session_converged(session) is True


def test_cumulative_not_fit_across_turns():
    """NOT_FITs accumulate across turns the same way FITs do."""
    rows_t1 = _rows(
        n_fit=CONVERGENCE_MIN_FIT, n_not_fit=1, pk_start=1
    )
    rows_t2 = _rows(
        n_fit=0, n_not_fit=CONVERGENCE_MIN_NOT_FIT - 1, pk_start=500
    )
    session = _session(
        [
            _turn(precision=CONVERGENCE_PRECISION, turn_number=1, rows=rows_t1),
            _turn(precision=CONVERGENCE_PRECISION, turn_number=2, rows=rows_t2),
        ]
    )
    assert is_session_converged(session) is True


def test_discard_rows_excluded_from_cumulative_pools():
    """DISCARD chunks count toward neither FIT nor NOT_FIT cumulative
    pools — they are an escape hatch, not a verdict (issue #46).
    They are exposed via a dedicated `_cumulative_discard_pks` helper so
    Phase 2 (anchor harvest) and Phase 3 (refine sampler) can drop them
    from downstream cohorts."""
    from src.replay.metrics import (
        _cumulative_discard_pks,
        _cumulative_fit_pks,
        _cumulative_not_fit_pks,
    )

    rows = _rows(n_fit=2, n_not_fit=2, pk_start=1)
    discard_row = EvidenceRow(
        rank=99,
        pk=999,
        chunk_id="C0000999",
        chunk_content="...",
        sample_id="S_x",
        counselor_id="C-1",
        term="t",
        source_paths=["dense"],
        scores={"dense": 0.5, "sparse": 0.0, "bm25": 0.0},
        rating="DISCARD",
    )
    rows.append(discard_row)
    session = _session([_turn(precision=0.5, turn_number=1, rows=rows)])

    fit_pks = _cumulative_fit_pks(session)
    not_fit_pks = _cumulative_not_fit_pks(session)
    discard_pks = _cumulative_discard_pks(session)
    assert 999 not in fit_pks
    assert 999 not in not_fit_pks
    assert 999 in discard_pks
    assert len(fit_pks) == 2
    assert len(not_fit_pks) == 2
    assert len(discard_pks) == 1


def test_duplicate_fit_pks_do_not_double_count():
    rows_t1 = _rows(
        n_fit=CONVERGENCE_MIN_FIT - 1,
        n_not_fit=CONVERGENCE_MIN_NOT_FIT,
        pk_start=1,
    )
    rows_t2 = _rows(
        n_fit=CONVERGENCE_MIN_FIT - 1,
        n_not_fit=CONVERGENCE_MIN_NOT_FIT,
        pk_start=1,
    )
    session = _session(
        [
            _turn(precision=CONVERGENCE_PRECISION, turn_number=1, rows=rows_t1),
            _turn(precision=CONVERGENCE_PRECISION, turn_number=2, rows=rows_t2),
        ]
    )
    assert is_session_converged(session) is False
