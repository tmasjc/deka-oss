"""In-memory session state for the tuning loop.

Tracks turn history, precision trends, convergence, and the
audit-mode + path-drop flow. Pure Python dataclasses consumed by the
web API and core modules — no UI framework dependencies.

Reflection no longer prescribes config changes: the session config is
locked once turn 1 starts. The only mid-session config change is a
path drop performed via :meth:`SessionState.apply_path_drop`, which
fires Rule B (sole-source FIT in the fused top-K, FIT among the
dropped path's per-path candidates) against the in-progress audit
turn's evidence.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.replay.metrics import (
    CONVERGENCE_MIN_FIT,
    CONVERGENCE_MIN_NOT_FIT,
    CONVERGENCE_PRECISION,
)
from src.search.config import with_overrides
from src.search.evidence import (
    EvidenceTable,
    PathName,
    PrimaryKey,
    Rating,
    compute_breakdown,
)


@dataclass(frozen=True)
class ConvergenceThresholds:
    """Per-session Phase 1 convergence triple, frozen at session start.

    Resolved from ``harvest.{min_fit, min_not_fit, precision_at_k}`` in
    ``config.yaml`` plus any per-session overrides. Locked alongside the
    search config so a YAML edit mid-session can't shift the gate under
    a running operator.
    """

    min_fit: int
    min_not_fit: int
    precision_at_k: float

    @classmethod
    def from_module_defaults(cls) -> "ConvergenceThresholds":
        return cls(
            min_fit=CONVERGENCE_MIN_FIT,
            min_not_fit=CONVERGENCE_MIN_NOT_FIT,
            precision_at_k=CONVERGENCE_PRECISION,
        )


if TYPE_CHECKING:
    from src.search.config import SearchConfig


# Phases drive app control flow; ``progress_state`` remains the display
# surface for the convergence gate. A session moves forward through the
# tuple below — once in a terminal anchor phase it stays there.
#
# Phase 3 (refine) phases stack on top of ANCHOR_DONE: the operator
# either ends the session (DONE) or accepts the refine offer, which
# walks REFINE_DERIVING → REFINE_EDITING → REFINE_JUDGING →
# REFINE_REVIEW → DONE (or back to REFINE_EDITING when the operator
# discards the judged round). The flow is gated by ``refine.enabled``
# in config.
#
# Phase 4 (apply) phases stack on top of REFINE_REVIEW finalise. When
# ``apply.enabled`` is true and Phase 3 finalises, the operator is
# offered a cohort-apply pass: APPLY_CONFIRM → APPLY_TRAINING →
# APPLY_PREPARING (cohort embeddings + initial projection are warmed
# in the same worker, so the operator never sees an empty panel) →
# APPLY_REVIEW (interactive threshold pick) → APPLY_APPLYING → DONE.
# APPLY_REVIEW is re-entrant — slider drags re-call the calibrate
# helper but stay in the same phase until finalize.
_VALID_PHASES = frozenset(
    {
        "TUNING",
        "ANCHOR_CONFIRM",
        "ANCHOR_RUNNING",
        "ANCHOR_DONE",
        "ANCHOR_FAILED",
        "REFINE_CONFIRM",
        "REFINE_DERIVING",
        "REFINE_EDITING",
        "REFINE_JUDGING",
        "REFINE_REVIEW",
        "REFINE_FAILED",
        "APPLY_CONFIRM",
        "APPLY_TRAINING",
        "APPLY_PREPARING",
        "APPLY_REVIEW",
        "APPLY_APPLYING",
        "APPLY_FAILED",
        "DONE",
    }
)


@dataclass
class TurnRecord:
    """Immutable snapshot of one completed turn."""

    turn_number: int
    query: str
    config: "SearchConfig"
    evidence_table: EvidenceTable
    breakdown: dict[str, dict[str, int]]
    precision: float
    reflection: dict[str, Any] | None = None
    # True iff the operator triggered a path audit during this turn —
    # i.e. per-path candidates were surfaced for rating, Rule B was
    # evaluated, and (possibly) a path drop was applied. Default False
    # for regular turns. Persisted to the canonical progress log.
    audit_turn: bool = False


@dataclass
class SessionState:
    """Mutable session state threaded through the turn loop."""

    current_config: "SearchConfig"
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    query: str | None = None
    turns: list[TurnRecord] = field(default_factory=list)
    current_table: EvidenceTable | None = None
    seen_pks: set[PrimaryKey] = field(default_factory=set)
    phase: str = "TUNING"
    scope: str | None = None
    # Audit mode is per-turn: it flips to True when the operator
    # surfaces per-path candidates for rating, and resets to False when
    # ``complete_turn`` runs. The agent's ``path_drop_recommendation``
    # no longer touches this flag — its ``[a]pply`` path drops directly
    # via :meth:`apply_recommended_drop`, bypassing the audit step.
    audit_mode_active: bool = False
    # One-shot session-level safeguard: once an agent recommendation
    # has been successfully applied (operator pressed ``[a]pply`` and
    # the drop landed), no further ``path_drop_recommendation`` from
    # the reflection agent is surfaced for the rest of the session.
    # Manual audit drops do NOT set this flag — they're
    # operator-initiated and outside the recommendation contract.
    recommended_drop_applied: bool = False
    # Phase 3 cross-screen state. Held as ``Any`` to keep state.py free
    # of refine imports — the web API sets this to a
    # :class:`src.refine.runner.RefineRunState` once the operator
    # accepts the refine offer.
    refine_state: Any = None
    # Phase 4 cross-screen state — same opaque-Any pattern as
    # ``refine_state``. Set to a :class:`src.apply.runner.ApplyRunState`
    # after ``run_apply_train`` completes; remains None until the
    # operator accepts the apply offer.
    apply_state: Any = None
    # Turn-0 probe summary (JSON-safe dict) used by the web UI to render
    # the probe banner. Populated by the web API after ``probe_only`` +
    # ``adapt_config``.
    probe_summary: dict[str, Any] | None = None
    # Phase 1 convergence thresholds locked at session start (YAML +
    # ``[Edit parameters]`` override). Reads here instead of the module
    # constants so a config edit or session override flows into both the
    # gate predicate and the Convergence panel display. Defaults to the
    # module constants so headless callers / tests that build a bare
    # ``SessionState()`` keep their existing behaviour.
    convergence: ConvergenceThresholds = field(
        default_factory=ConvergenceThresholds.from_module_defaults
    )

    # -- derived properties ------------------------------------------------

    @property
    def turn_number(self) -> int:
        return len(self.turns) + 1

    @property
    def precision_trend(self) -> list[float]:
        return [t.precision for t in self.turns]

    @property
    def cumulative_fit_pks(self) -> set[PrimaryKey]:
        """Unique PKs rated FIT across all completed turns.

        Counts both fused-row ratings and per-path candidate ratings,
        deduped by PK so a chunk surfaced in multiple turns counts once.
        On regular (non-audit) turns the per-path candidates are never
        rated — so they contribute nothing — but the union shape stays
        the same to keep the convergence predicate honest across turn
        kinds.
        """
        fit: set[PrimaryKey] = set()
        for turn in self.turns:
            for row in turn.evidence_table.rows:
                if row.rating == "FIT":
                    fit.add(row.pk)
            for candidates in turn.evidence_table.per_path_candidates.values():
                for cand in candidates:
                    if cand.rating == "FIT":
                        fit.add(cand.pk)
        return fit

    @property
    def cumulative_not_fit_pks(self) -> set[PrimaryKey]:
        """Unique PKs rated NOT_FIT across all completed turns.

        Mirror of :attr:`cumulative_fit_pks` for the contrastive side
        of the convergence gate. Phase 3's rubric deriver needs
        negative examples to discriminate; the gate's job is to confirm
        the operator surfaced and rejected enough of them.
        """
        not_fit: set[PrimaryKey] = set()
        for turn in self.turns:
            for row in turn.evidence_table.rows:
                if row.rating == "NOT_FIT":
                    not_fit.add(row.pk)
            for candidates in turn.evidence_table.per_path_candidates.values():
                for cand in candidates:
                    if cand.rating == "NOT_FIT":
                        not_fit.add(cand.pk)
        return not_fit

    @property
    def is_converged(self) -> bool:
        """Latest turn P@K >= ``convergence.precision_at_k`` AND cumulative
        unique FIT PKs >= ``convergence.min_fit`` AND cumulative unique
        NOT_FIT PKs >= ``convergence.min_not_fit``. Thresholds are
        snapshotted at session creation from ``harvest.*`` + overrides."""
        if not self.turns:
            return False
        latest = self.turns[-1].precision
        return (
            latest >= self.convergence.precision_at_k
            and len(self.cumulative_fit_pks) >= self.convergence.min_fit
            and len(self.cumulative_not_fit_pks) >= self.convergence.min_not_fit
        )

    @property
    def progress_state(self) -> str:
        """'START' before any turn completes, 'TURN_N' after turn N,
        'CONVERGED' once convergence criteria hold.
        """
        if self.is_converged:
            return "CONVERGED"
        if not self.turns:
            return "START"
        return f"TURN_{len(self.turns)}"

    # -- mutations ---------------------------------------------------------

    def start_turn(self, table: EvidenceTable) -> None:
        self.current_table = table
        if self.query is None:
            self.query = table.query

    def rate(self, rank: int, rating: Rating) -> None:
        if self.current_table is None:
            raise RuntimeError("No active turn — call start_turn first")
        self.current_table.set_rating(rank, rating)

    def rate_candidate(self, path: str, rank_in_path: int, rating: Rating) -> None:
        if self.current_table is None:
            raise RuntimeError("No active turn — call start_turn first")
        self.current_table.set_candidate_rating(path, rank_in_path, rating)  # type: ignore[arg-type]

    def all_rated(self) -> bool:
        """True iff the fused top-K (and, in audit mode, also the
        per-path candidates) are fully rated."""
        if self.current_table is None:
            return False
        return self.current_table.all_rated(include_candidates=self.audit_mode_active)

    def apply_path_drop(self, path: str) -> tuple[bool, str | None]:
        """Apply a path drop to ``current_config`` after Rule B clears.

        Operates on ``self.current_table`` — the in-progress audit turn's
        evidence — so the operator's drop decision is anchored to the
        ratings they just produced. Returns ``(allowed, error)``:

        - ``(True, None)``: drop is applied; ``current_config.active_paths``
          is updated for the remainder of the session.
        - ``(False, message)``: Rule B blocked the drop; the session
          config is unchanged. The message names the path and the reason
          (sole-source FIT row(s), or FIT candidate ratings on the
          dropped path).

        Re-activation of a previously-dropped path is not handled here —
        that's a session-start operation, never mid-session.
        """
        if self.current_table is None:
            return False, "No turn in progress to audit."

        if path not in self.current_config.active_paths:
            return False, f"{path!r} is not currently active."

        if len(self.current_config.active_paths) <= 1:
            return False, "Cannot drop the last active path."

        # Rule B1 — fused sole source: any FIT row whose only source is
        # the path being dropped is guaranteed lost.
        sole_source_fit = [
            row
            for row in self.current_table.rows
            if row.rating == "FIT" and row.source_paths == [path]
        ]
        if sole_source_fit:
            ranks = ", ".join(str(r.rank) for r in sole_source_fit)
            return False, (
                f"Cannot deactivate {path}: fused top-K contains "
                f"{len(sole_source_fit)} FIT chunk(s) sourced only from "
                f"{path} (rank{'s' if len(sole_source_fit) > 1 else ''} "
                f"{ranks}). These would be lost."
            )

        # Rule B2 — candidate FIT: a path with FIT among its per-path
        # candidates has demonstrated value outside the fused set.
        candidates = self.current_table.per_path_candidates.get(path, [])  # type: ignore[arg-type]
        rated = [c for c in candidates if c.rating is not None]
        fit_count = sum(1 for c in rated if c.rating == "FIT")
        if fit_count >= 1:
            return False, (
                f"Cannot deactivate {path}: its per-path candidates include "
                f"FIT results ({fit_count}/{len(rated)} FIT). A path with "
                "evidence of retrieving relevant material should not be dropped."
            )

        new_active = frozenset(self.current_config.active_paths - {path})
        self.current_config = with_overrides(
            self.current_config, active_paths=new_active
        )
        return True, None

    def apply_recommended_drop(self, path: str) -> tuple[bool, str | None]:
        """Apply an agent-recommended path drop directly.

        This is the apply target for the structured
        ``path_drop_recommendation`` flow. Skips Rule B entirely —
        the agent is expected to have weighed Rule B1 from the
        progress log before populating the recommendation. The
        operator's ``[a]pply`` decision is the gate; the apply itself
        is mechanical.

        Only the universal mechanical guards apply:

        - The path must be currently active.
        - The drop must not leave ``active_paths`` empty.

        Returns ``(True, None)`` on success and updates
        ``current_config.active_paths``. Returns ``(False, reason)``
        when a guard rejects, with ``current_config`` unchanged.

        For operator-initiated audits, use :meth:`apply_path_drop`
        instead — it enforces Rule B1 and B2 against the in-progress
        audit turn's evidence.
        """
        if path not in self.current_config.active_paths:
            return False, f"{path!r} is not currently active."

        if len(self.current_config.active_paths) <= 1:
            return False, "Cannot drop the last active path."

        new_active = frozenset(self.current_config.active_paths - {path})
        self.current_config = with_overrides(
            self.current_config, active_paths=new_active
        )
        return True, None

    def complete_turn(self) -> TurnRecord:
        if self.current_table is None:
            raise RuntimeError("No active turn to complete")
        if not self.all_rated():
            raise RuntimeError("Cannot complete turn with unrated rows")

        breakdown = compute_breakdown(self.current_table)
        fit_count = sum(1 for r in self.current_table.rows if r.rating == "FIT")
        not_fit_count = sum(1 for r in self.current_table.rows if r.rating == "NOT_FIT")
        # DISCARD chunks are an escape hatch (issue #46), not a verdict — they
        # don't dilute the precision signal.
        verdict_total = fit_count + not_fit_count
        precision = fit_count / verdict_total if verdict_total > 0 else 0.0

        record = TurnRecord(
            turn_number=self.turn_number,
            query=self.current_table.query,
            config=self.current_config,
            evidence_table=self.current_table,
            breakdown=breakdown,
            precision=precision,
            audit_turn=self.audit_mode_active,
        )
        # Union the PKs the human saw this turn into the session-wide seen
        # set so subsequent searches exclude them via Milvus ``expr``. This
        # covers both fused rows and per-path top-3 candidates — both have
        # been fetched from Milvus this turn even when only fused rows
        # were rated, so neither should resurface unbidden later.
        turn_pks = {row.pk for row in self.current_table.rows}
        for candidates in self.current_table.per_path_candidates.values():
            for cand in candidates:
                turn_pks.add(cand.pk)
        self.turns.append(record)
        self.seen_pks.update(turn_pks)
        self.current_table = None
        self.audit_mode_active = False
        return record

    def set_next_config(self, config: "SearchConfig") -> None:
        self.current_config = config

    def set_phase(self, phase: str) -> None:
        if phase not in _VALID_PHASES:
            raise ValueError(
                f"Invalid phase {phase!r}; expected one of {sorted(_VALID_PHASES)}"
            )
        self.phase = phase


__all__ = [
    "CONVERGENCE_MIN_FIT",
    "CONVERGENCE_MIN_NOT_FIT",
    "CONVERGENCE_PRECISION",
    "ConvergenceThresholds",
    "PathName",
    "SessionState",
    "TurnRecord",
]
