"""Tests for the right-panel workflow timeline in ``src.web_api.serialize``.

The pending "TURN N+1" step injected during the tuning loop must be
gated on ``state.phase == "TUNING"``, not on ``state.is_converged``.
The operator-override-low-precision path advances past TUNING without
the convergence predicate ever flipping True; gating on the metric
caused a phantom pending turn to render alongside the already-done
CONVERGED / HARVEST nodes whenever such a session was replayed (or
viewed live post-harvest).
"""

from __future__ import annotations

from src.search.config import SearchConfig
from src.search.evidence import EvidenceTable
from src.session.state import SessionState, TurnRecord
from src.web_api.serialize import workflow_steps


def _config() -> SearchConfig:
    return SearchConfig(rrf_k=60, per_path_limit=20, top_k=2)


def _empty_table() -> EvidenceTable:
    return EvidenceTable(query="q", config=_config(), rows=[])


def _state_with_turns(phase: str, precisions: list[float]) -> SessionState:
    state = SessionState(current_config=_config())
    state.set_phase(phase)
    for i, p in enumerate(precisions, start=1):
        state.turns.append(
            TurnRecord(
                turn_number=i,
                query="q",
                config=_config(),
                evidence_table=_empty_table(),
                breakdown={},
                precision=p,
            )
        )
    return state


def test_pending_turn_injected_at_start_of_tuning() -> None:
    state = _state_with_turns("TUNING", precisions=[])
    keys = [s.key for s in workflow_steps(state)]
    assert keys == ["START", "TURN_1", "CONVERGED", "HARVEST"]
    pending = next(s for s in workflow_steps(state) if s.key == "TURN_1")
    assert pending.status == "current"
    assert pending.detail == "pending"


def test_pending_turn_injected_after_completed_turn_still_tuning() -> None:
    state = _state_with_turns("TUNING", precisions=[0.50])
    steps = workflow_steps(state)
    keys = [s.key for s in steps]
    assert keys == ["START", "TURN_1", "TURN_2", "CONVERGED", "HARVEST"]
    done = next(s for s in steps if s.key == "TURN_1")
    pending = next(s for s in steps if s.key == "TURN_2")
    assert done.status == "done"
    assert pending.status == "current"
    assert pending.detail == "pending"


def test_no_phantom_pending_turn_when_phase_past_tuning() -> None:
    """Operator-override path: phase is past TUNING but is_converged is False.

    Regression for the replay-mode bug where a completed session
    rendered "TURN N+1 pending" alongside the green CONVERGED through
    SHIP nodes. The pending step must not appear once phase has moved
    on, regardless of the convergence metric.
    """
    state = _state_with_turns("DONE", precisions=[0.50])
    assert not state.is_converged, (
        "fixture must reproduce the override case (low P@K, no FITs)"
    )
    keys = [s.key for s in workflow_steps(state)]
    assert "TURN_1" in keys
    assert "TURN_2" not in keys, "no phantom pending turn when phase is past TUNING"


def test_no_phantom_pending_in_anchor_phase() -> None:
    """During ANCHOR_RUNNING the pending step must not be injected even
    if is_converged is False (operator-override path mid-harvest).
    """
    state = _state_with_turns("ANCHOR_RUNNING", precisions=[0.50])
    keys = [s.key for s in workflow_steps(state)]
    assert "TURN_2" not in keys
