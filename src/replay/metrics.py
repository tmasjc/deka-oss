"""Headless metrics on a :class:`ReplaySession`.

Canonical home for the convergence rule shared by the live session
(``src.session.state.SessionState``) and the headless sweep CLI.
Convergence thresholds are part of the unified ``harvest:`` YAML
section (see :mod:`src.anchor.config`) — Phase 1 convergence and the
Phase 2 retrieval defaults share one home because the whole Phase 1 →
Phase 2 hand-off is one lifecycle.
"""

from __future__ import annotations

from src.anchor.config import load_harvest_config
from src.search.evidence import PrimaryKey

from .loader import ReplaySession


# Loaded at import time so the module-level constants stay plain
# scalars. Tests that need a different value can monkeypatch the
# attribute; the YAML is the source of truth for every other caller.
_CFG = load_harvest_config()
CONVERGENCE_MIN_FIT = _CFG.min_fit
CONVERGENCE_MIN_NOT_FIT = _CFG.min_not_fit
CONVERGENCE_PRECISION = _CFG.precision_at_k


def _eval_gate(
    latest_precision: float, cumulative_fit: int, cumulative_not_fit: int
) -> bool:
    """Shared triple-gate predicate — the one definition of convergence."""

    return (
        latest_precision >= CONVERGENCE_PRECISION
        and cumulative_fit >= CONVERGENCE_MIN_FIT
        and cumulative_not_fit >= CONVERGENCE_MIN_NOT_FIT
    )


def _cumulative_fit_pks(session: ReplaySession) -> set[PrimaryKey]:
    """Union of FIT-rated PKs across every turn, deduped.

    Counts both fused-row ratings and per-path candidate ratings so a
    chunk surfaced in multiple turns counts once. Mirrors
    :meth:`src.session.state.SessionState.cumulative_fit_pks` for a
    replayed session.
    """

    fit: set[PrimaryKey] = set()
    for turn in session.turns:
        for row in turn.evidence_table.rows:
            if row.rating == "FIT":
                fit.add(row.pk)
        for candidates in turn.evidence_table.per_path_candidates.values():
            for cand in candidates:
                if cand.rating == "FIT":
                    fit.add(cand.pk)
    return fit


def _cumulative_not_fit_pks(session: ReplaySession) -> set[PrimaryKey]:
    """Union of NOT_FIT-rated PKs across every turn, deduped.

    Mirror of :func:`_cumulative_fit_pks` for the contrastive side of
    the convergence gate. Last-write-wins is *not* applied here — a pk
    rated NOT_FIT in any turn counts toward the floor even if a later
    turn flipped it to FIT. The gate's purpose is to confirm the
    operator saw and rejected enough negatives for Phase 3 to derive a
    discriminator; flips are rare and a stricter same-pk policy would
    needlessly fight the gate.
    """

    not_fit: set[PrimaryKey] = set()
    for turn in session.turns:
        for row in turn.evidence_table.rows:
            if row.rating == "NOT_FIT":
                not_fit.add(row.pk)
        for candidates in turn.evidence_table.per_path_candidates.values():
            for cand in candidates:
                if cand.rating == "NOT_FIT":
                    not_fit.add(cand.pk)
    return not_fit


def _cumulative_discard_pks(session: ReplaySession) -> set[PrimaryKey]:
    """Union of DISCARD-rated PKs across every turn, deduped.

    DISCARD is an operator-issued invalidation (issue #46): chunks with
    obvious defects (STT garble, broken grammar, off-topic content)
    that should never propagate downstream. Phase 2 uses this set to
    filter the harvest output; Phase 3 uses it to keep DISCARD chunks
    out of the judge sample.
    """

    discard: set[PrimaryKey] = set()
    for turn in session.turns:
        for row in turn.evidence_table.rows:
            if row.rating == "DISCARD":
                discard.add(row.pk)
        for candidates in turn.evidence_table.per_path_candidates.values():
            for cand in candidates:
                if cand.rating == "DISCARD":
                    discard.add(cand.pk)
    return discard


def is_session_converged(session: ReplaySession) -> bool:
    """True iff the triple convergence gate is satisfied.

    Triple-gate definition shared with
    :attr:`src.session.state.SessionState.is_converged`: the latest turn's
    Precision@K meets ``CONVERGENCE_PRECISION``, the cumulative unique
    FIT count across all turns meets ``CONVERGENCE_MIN_FIT``, and the
    cumulative unique NOT_FIT count meets ``CONVERGENCE_MIN_NOT_FIT``.
    A session with no turns is not converged.
    """

    if not session.turns:
        return False
    latest_precision = session.turns[-1].precision
    cumulative_fit = len(_cumulative_fit_pks(session))
    cumulative_not_fit = len(_cumulative_not_fit_pks(session))
    return _eval_gate(latest_precision, cumulative_fit, cumulative_not_fit)
