"""Resume-target classification + per-target session hydration.

A session's resume target is a pure function of which sidecars exist
under ``runs/<user_id>/<sid>.*``. The classifier reads only the
file system — no LLM, no Milvus, no JSONL parsing beyond the last
line of the canonical log — so it's cheap enough to call once per
listed session in the ``GET /api/sessions`` endpoint.

Branch order (each only reached when the earlier ones don't match):

1. ``phase3.meta.json`` exists with ``operator_decision == "agree"``
   → :attr:`ResumeTarget.DONE_VIEW`.
2. ``phase3.rubric.json`` and ``phase3.evidence.jsonl`` exist
   → :attr:`ResumeTarget.POST_RUBRIC`.
3. ``phase2.meta.json`` exists → :attr:`ResumeTarget.POST_HARVEST`.
4. The canonical JSONL's last line is the convergence marker
   (``{"event": "converged", ...}``) →
   :attr:`ResumeTarget.POST_TUNING`.
5. Otherwise the session is **abandoned** (returns ``None``).

The two-stage Phase 3 writer (PR1) and the Phase 1 convergence
marker (PR1) make this branching deterministic — anything else
would need parsing the canonical log front-to-back per session,
which doesn't scale to a listing endpoint.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.anchor.config import load_harvest_config
from src.anchor.errors import AnchorError
from src.replay.loader import ReplayLoadError, ReplaySession, load_session
from src.session.state import ConvergenceThresholds, SessionState, TurnRecord

if TYPE_CHECKING:
    from src.web_api.sessions import SessionContext, SessionStore

log = logging.getLogger(__name__)


class ResumeError(RuntimeError):
    """Raised when an on-disk session cannot be hydrated.

    Distinct from a missing session — used by the resume endpoint to
    distinguish "session is on disk but can't be cleanly rebuilt" (a
    422-class problem) from "session classifies abandoned" (a 404).
    """


class ResumeNotImplementedError(ResumeError):
    """Raised when hydration for this resume target isn't wired up yet.

    Surfaced by the resume endpoint as 501 with a clear detail message.
    Covers POST_HARVEST / POST_RUBRIC / DONE_VIEW until full state
    reconstruction lands in a follow-up.
    """


# Every sidecar suffix a single session may produce. Used by
# :func:`_session_last_modified` to compute the listing's sort key.
_SESSION_SUFFIXES: tuple[str, ...] = (
    ".jsonl",
    ".details.jsonl",
    ".log",
    ".span_cache.jsonl",
    ".phase2.jsonl",
    ".phase2.meta.json",
    ".phase3.prompt.md",
    ".phase3.rubric.json",
    ".phase3.evidence.jsonl",
    ".phase3.meta.json",
    ".owner",
    ".overrides.json",
)


class ResumeTarget(str, Enum):
    """Where a returning user lands after clicking a session row."""

    POST_TUNING = "POST_TUNING"
    POST_HARVEST = "POST_HARVEST"
    POST_RUBRIC = "POST_RUBRIC"
    APPLY_PENDING = "APPLY_PENDING"
    DONE_VIEW = "DONE_VIEW"


def _read_last_line(path: Path, *, tail_bytes: int = 8192) -> str | None:
    """Return the file's last non-empty line, or ``None`` if empty.

    Reads from the tail (``tail_bytes`` window) to avoid loading
    multi-megabyte canonical logs in full. Falls back to reading the
    whole file when the tail window contains no newline (file shorter
    than the window).
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size == 0:
        return None
    try:
        with path.open("rb") as fp:
            if size <= tail_bytes:
                fp.seek(0)
                buf = fp.read()
            else:
                fp.seek(size - tail_bytes, os.SEEK_SET)
                buf = fp.read()
    except OSError as exc:
        log.warning("classify: could not read tail of %s: %s", path, exc)
        return None
    text = buf.decode("utf-8", errors="replace")
    # Strip trailing newline / whitespace, then take the last line.
    text = text.rstrip("\r\n").rstrip()
    if not text:
        return None
    return text.rsplit("\n", 1)[-1]


def classify(sid: str, user_runs_dir: Path) -> ResumeTarget | None:
    """Decide where a session resumes to, or ``None`` if abandoned.

    Pure: only reads the filesystem. Safe to call from a listing
    endpoint without locking. ``user_runs_dir`` is expected to be the
    ``runs/<user_id>/`` path returned by
    :meth:`SessionStore.runs_dir_for`.
    """
    meta_p3 = user_runs_dir / f"{sid}.phase3.meta.json"
    if meta_p3.exists():
        try:
            payload = json.loads(meta_p3.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("classify(%s): unreadable phase3.meta.json: %s", sid, exc)
            payload = None
        if isinstance(payload, dict) and payload.get("operator_decision") == "agree":
            # Phase 3 finalised. If Phase 4 is enabled and labels.jsonl
            # hasn't shipped yet, the session is still actionable —
            # signal that to the list UI so it doesn't badge "Done"
            # prematurely.
            if _apply_pending(sid, user_runs_dir):
                return ResumeTarget.APPLY_PENDING
            return ResumeTarget.DONE_VIEW

    rubric_p3 = user_runs_dir / f"{sid}.phase3.rubric.json"
    evidence_p3 = user_runs_dir / f"{sid}.phase3.evidence.jsonl"
    if rubric_p3.exists() and evidence_p3.exists():
        return ResumeTarget.POST_RUBRIC

    meta_p2 = user_runs_dir / f"{sid}.phase2.meta.json"
    if meta_p2.exists():
        return ResumeTarget.POST_HARVEST

    canonical = user_runs_dir / f"{sid}.jsonl"
    if canonical.exists():
        last = _read_last_line(canonical)
        if last is not None:
            try:
                row = json.loads(last)
            except json.JSONDecodeError:
                row = None
            if isinstance(row, dict) and row.get("event") == "converged":
                return ResumeTarget.POST_TUNING

    return None


@dataclass(frozen=True)
class SessionListEntry:
    """One row in the session-picker listing.

    ``scope`` is ``None`` only for sessions whose canonical jsonl
    has no readable turn rows yet (rare — those usually classify as
    abandoned and are excluded from the listing).
    """

    session_id: str
    query: str
    scope: str | None
    resume_target: ResumeTarget
    last_modified: str  # ISO 8601, UTC, second resolution
    n_turns: int


def _scan_canonical_for_listing(
    canonical_path: Path,
) -> tuple[dict | None, int]:
    """Single pass: return (first turn row, count of turn rows).

    Skips event rows (those carrying an ``event`` key — see
    ``harness/schemas/progress_log.md``). Returns ``(None, 0)`` if the
    file is missing or empty.
    """
    if not canonical_path.exists():
        return None, 0
    first_turn: dict | None = None
    n_turns = 0
    try:
        with canonical_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if "event" in row:
                    continue
                if first_turn is None:
                    first_turn = row
                n_turns += 1
    except OSError as exc:
        log.warning("listing scan failed for %s: %s", canonical_path, exc)
    return first_turn, n_turns


def _session_last_modified(sid: str, user_runs_dir: Path) -> float:
    """Highest mtime across this session's sidecars, in epoch seconds.

    Used as the sort key for the listing. Returns 0.0 if no sidecars
    are readable (caller filters such sessions out via ``classify``).
    """
    latest = 0.0
    for suffix in _SESSION_SUFFIXES:
        path = user_runs_dir / f"{sid}{suffix}"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > latest:
            latest = mtime
    return latest


def _iso(epoch: float) -> str:
    if epoch <= 0.0:
        return "1970-01-01T00:00:00Z"
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _session_ids_in(user_runs_dir: Path) -> list[str]:
    """Return the session ids visible under ``user_runs_dir``.

    A session id is the prefix of any ``<sid>.jsonl`` or
    ``<sid>.owner`` file (the two markers most reliably present at
    create-time). Phase 2 / Phase 3 sidecars alone do not seed the
    list — they always coexist with the canonical.
    """
    if not user_runs_dir.exists():
        return []
    sids: set[str] = set()
    for entry in user_runs_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(".owner"):
            sids.add(name[: -len(".owner")])
        elif name.endswith(".jsonl") and not (
            name.endswith(".details.jsonl")
            or name.endswith(".phase2.jsonl")
            or name.endswith(".phase3.evidence.jsonl")
            or name.endswith(".span_cache.jsonl")
        ):
            sids.add(name[: -len(".jsonl")])
    return sorted(sids)


def list_user_sessions(user_runs_dir: Path) -> list[SessionListEntry]:
    """Build the post-login session-picker payload.

    Walks ``runs/<user_id>/``, classifies each session, drops any
    that classify as abandoned, and returns the rest sorted by
    ``last_modified`` descending. Pure: filesystem reads only.
    """
    entries: list[SessionListEntry] = []
    for sid in _session_ids_in(user_runs_dir):
        target = classify(sid, user_runs_dir)
        if target is None:
            continue
        canonical = user_runs_dir / f"{sid}.jsonl"
        first_turn, n_turns = _scan_canonical_for_listing(canonical)
        query = ""
        scope: str | None = None
        if first_turn is not None:
            raw_query = first_turn.get("query")
            if isinstance(raw_query, str):
                query = raw_query
            raw_scope = first_turn.get("scope")
            if isinstance(raw_scope, str):
                scope = raw_scope
        last_modified = _iso(_session_last_modified(sid, user_runs_dir))
        entries.append(
            SessionListEntry(
                session_id=sid,
                query=query,
                scope=scope,
                resume_target=target,
                last_modified=last_modified,
                n_turns=n_turns,
            )
        )
    entries.sort(key=lambda e: e.last_modified, reverse=True)
    return entries


def _build_state_from_replay(
    replay: ReplaySession,
    *,
    scope: str | None,
    user_runs_dir: Path | None = None,
) -> SessionState:
    """Rebuild a mutable :class:`SessionState` from a frozen
    :class:`ReplaySession`.

    Recovers everything the harvest-confirm modal and the live
    rating loop need: full turn history, the locked seed config
    (taken from the last turn), seen_pks, query, scope. The
    convergence predicate (``state.is_converged``) re-derives from
    the reconstructed turn metrics — no separate flag to set.

    When ``user_runs_dir`` is supplied the function also resolves the
    session's locked convergence thresholds from its on-disk overrides
    sidecar + current YAML, so a resumed session displays the same
    ``FIT ≥ N`` / ``NOT_FIT ≥ M`` rule it ran under. Without
    ``user_runs_dir`` (legacy callers) the state falls back to the
    module-default thresholds.

    ``audit_turn`` on each rebuilt :class:`TurnRecord` defaults to
    ``False``: it's a historical bookkeeping flag, irrelevant to
    further operations on the resumed session.
    """
    if not replay.turns:
        raise ResumeError(
            f"Session {replay.session_id} has no turn rows; cannot hydrate"
        )

    last_turn = replay.turns[-1]
    convergence = _convergence_from_disk(replay.session_id, user_runs_dir)
    state = SessionState(
        current_config=last_turn.config,
        session_id=replay.session_id,
        convergence=convergence,
    )
    state.scope = scope
    state.query = last_turn.query

    for replay_turn in replay.turns:
        record = TurnRecord(
            turn_number=replay_turn.turn_number,
            query=replay_turn.query,
            config=replay_turn.config,
            evidence_table=replay_turn.evidence_table,
            breakdown=replay_turn.breakdown,
            precision=replay_turn.precision,
            reflection=replay_turn.reflection,
            audit_turn=False,
        )
        state.turns.append(record)
        for row in replay_turn.evidence_table.rows:
            state.seen_pks.add(row.pk)
        for candidates in replay_turn.evidence_table.per_path_candidates.values():
            for cand in candidates:
                state.seen_pks.add(cand.pk)

    return state


def _read_scope_from_canonical(canonical_path: Path) -> str | None:
    """Best-effort scope lookup from the first turn row of the canonical
    log. Returns ``None`` if the file is missing or carries no
    ``scope`` field (older sessions, or replay-loaded states).
    """
    if not canonical_path.exists():
        return None
    try:
        with canonical_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if "event" in row:
                    continue
                value = row.get("scope")
                if isinstance(value, str) and value.strip():
                    return value
                return None
    except OSError:
        return None
    return None


def hydrate_post_tuning(
    sid: str,
    *,
    user_runs_dir: Path,
    store: "SessionStore",
    user_id: str,
) -> "SessionContext":
    """Rebuild a session frozen at the post-convergence / pre-harvest
    boundary.

    Reuses :func:`src.replay.loader.load_session` to read every turn
    from disk, then constructs a mutable :class:`SessionState` ready
    for the live harvest-confirm flow. Registers the context in the
    store via :meth:`SessionStore.attach` so subsequent
    ``/api/session/<sid>/...`` calls find it warm.
    """
    canonical = user_runs_dir / f"{sid}.jsonl"
    try:
        replay = load_session(canonical)
    except ReplayLoadError as exc:
        raise ResumeError(f"Could not load canonical log for {sid}: {exc}") from exc
    scope = _read_scope_from_canonical(canonical)
    state = _build_state_from_replay(
        replay, scope=scope, user_runs_dir=user_runs_dir
    )
    return store.attach(state, user_id=user_id, read_only=False)


def _load_replay(sid: str, user_runs_dir: Path) -> ReplaySession:
    canonical = user_runs_dir / f"{sid}.jsonl"
    try:
        return load_session(canonical)
    except ReplayLoadError as exc:
        raise ResumeError(f"Could not load canonical log for {sid}: {exc}") from exc


def _reconstruct_anchor_result(replay: ReplaySession):
    """Rebuild :class:`src.anchor.runner.AnchorResult` from on-disk
    Phase 2 sidecars.

    Live API calls post-resume read the result through
    :mod:`src.web_api.serialize` (``anchor_result_to_dto``,
    ``workflow_steps``) and through the harvest/refine endpoint guards;
    none of them touch ``inputs.fits`` (which would carry embeddings
    not persisted to disk), so it is reconstructed as an empty list.
    """
    from src.anchor.config import RadiusScheme
    from src.anchor.loader import AnchorInputs
    from src.anchor.retrieve import AnchorCandidate, PerFitPages, RetrievalResult
    from src.anchor.runner import (
        AnchorResult,
        AnchorTimings,
        FrequencyGateSummary,
    )
    from src.anchor.threshold import CalibrationResult
    from src.anchor.validate import LooPerFit, RecoveryResult
    from src.anchor.writer import WriteResult

    try:
        meta = replay.load_phase2_meta()
        kept_rows = replay.load_phase2_kept()
    except ReplayLoadError as exc:
        raise ResumeError(str(exc)) from exc

    inputs = AnchorInputs(
        session_id=str(meta.get("session_id", replay.session_id)),
        query=str(meta.get("query", "")),
        collection=str(meta.get("collection", "")),
        embed_url="",
        embed_model_id=str(meta.get("embed_model_id", "")),
        milvus_index_type=str(meta.get("milvus_index_type", "UNKNOWN")),
        milvus_index_params=dict(meta.get("milvus_index_params") or {}),
        fits=[],
        not_fit_pks=frozenset(),
        discard_pks=frozenset(),
    )

    calib_block = meta.get("calibration") or {}
    calibration = CalibrationResult(
        T=float(calib_block.get("T", 0.0)),
        deltas=[float(d) for d in calib_block.get("deltas") or []],
        T_primes=[float(t) for t in calib_block.get("T_primes") or []],
        T_prime_out=float(calib_block.get("T_prime_out", 0.0)),
        span_loo_distances=[
            float(d) for d in calib_block.get("span_loo_distances") or []
        ],
        n_fit=int(meta.get("n_fit", 0)),
    )

    loo_block = meta.get("loo_recovery") or {}
    missed_fits = [
        LooPerFit(
            fit_pk=item.get("fit_pk", ""),
            fit_chunk_id=str(item.get("fit_chunk_id", "")),
            recovered=False,
            recalibrated_T=0.0,
            rank_of_own_pk=None,
            distance_of_own_pk=None,
        )
        for item in loo_block.get("missed_fits") or []
    ]
    recovery = RecoveryResult(
        recovered=int(loo_block.get("recovered", 0)),
        total=int(loo_block.get("total", 0)),
        verdict=str(loo_block.get("verdict", "HEALTHY")),  # type: ignore[arg-type]
        per_fit=[],
        missed_fits=missed_fits,
    )

    candidates: list[AnchorCandidate] = [
        AnchorCandidate(
            pk=row["pk"],
            nearest_fit_pk=row.get("nearest_fit_pk", ""),
            nearest_fit_label=str(row.get("nearest_fit_label", "")),
            nearest_fit_distance=float(row.get("nearest_fit_distance", 0.0)),
            qualifying_fit_pks=tuple(row.get("qualifying_fit_pks") or ()),
            qualifying_fit_count=int(row.get("qualifying_fit_count", 0)),
        )
        for row in kept_rows
        if "pk" in row
    ]

    exhausted_chunk_ids = {
        item.get("fit_chunk_id")
        for item in (meta.get("per_fit_budget_exhausted") or [])
    }
    per_fit_pages: list[PerFitPages] = []
    for entry in meta.get("per_fit_pages") or []:
        chunk_id = str(entry.get("fit_chunk_id", ""))
        kth = entry.get("final_kth_distance")
        per_fit_pages.append(
            PerFitPages(
                fit_pk=entry.get("fit_pk", ""),
                fit_chunk_id=chunk_id,
                pages_fetched=int(entry.get("pages_fetched", 0)),
                total_hits=int(entry.get("total_hits", 0)),
                final_kth_distance=(float(kth) if kth is not None else float("inf")),
                budget_exhausted=chunk_id in exhausted_chunk_ids,
            )
        )

    retrieval = RetrievalResult(
        candidates=candidates,
        n_raw_hits=0,
        n_unique=len(candidates),
        batch_size=int(meta.get("batch_size", 0)),
        max_k=int(meta.get("max_k", 0)),
        per_fit_pages=per_fit_pages,
        retained_pks=frozenset(c.pk for c in candidates),
        qualifying_count_by_pk={c.pk: c.qualifying_fit_count for c in candidates},
    )

    write_result = WriteResult(
        jsonl_path=replay.phase2_jsonl_path,
        meta_path=replay.phase2_meta_path,
        details_path=replay.details_path,
        n_records=int(meta.get("output_count", len(candidates))),
    )

    timings_block = meta.get("timings") or {}
    timings = AnchorTimings(
        load_ms=float(timings_block.get("load_ms", 0.0)),
        calibrate_ms=float(timings_block.get("calibrate_ms", 0.0)),
        loo_ms=float(timings_block.get("loo_ms", 0.0)),
        retrieve_ms=float(timings_block.get("retrieve_ms", 0.0)),
        total_ms=float(timings_block.get("total_ms", 0.0)),
    )

    radius_scheme_str = calib_block.get("radius_scheme") or RadiusScheme.PER_FIT.value
    try:
        radius_scheme = RadiusScheme(radius_scheme_str)
    except ValueError:
        radius_scheme = RadiusScheme.PER_FIT

    quality_block = meta.get("quality_gate") or {}
    quality_dropped = [dict(rec) for rec in quality_block.get("dropped") or []]
    cohort = [dict(rec) for rec in meta.get("cohort_consistency") or []]

    # Re-derive the multiplier-rule regime so the UI shows it correctly
    # on replay (issue #47 follow-up). Post-PR-50 sidecars carry
    # ``multiplier_cutoff`` directly; legacy sidecars only carry the
    # inputs (``s2c_outlier_multiple`` + ``median_delta_pre_drop``), so
    # we recompute the cutoff the same way ``_apply_quality_gate``
    # would have. The 1e-3 floor mirrors ``_MEDIAN_DELTA_EPSILON`` in
    # ``src/anchor/runner.py``.
    quality_gate_multiplier = float(quality_block.get("s2c_outlier_multiple", 0.0))
    quality_gate_median_pre = float(quality_block.get("median_delta_pre_drop", 0.0))
    if "multiplier_cutoff" in quality_block:
        raw_cutoff = quality_block["multiplier_cutoff"]
        quality_gate_multiplier_cutoff: float | None = (
            float(raw_cutoff) if raw_cutoff is not None else None
        )
    elif quality_gate_median_pre > 1e-3 and quality_gate_multiplier > 0:
        quality_gate_multiplier_cutoff = quality_gate_multiplier * quality_gate_median_pre
    else:
        quality_gate_multiplier_cutoff = None

    fg_block = meta.get("frequency_gate")
    frequency_gate: FrequencyGateSummary | None = None
    if isinstance(fg_block, dict):
        frequency_gate = FrequencyGateSummary(
            f_configured=int(fg_block.get("f_configured", 1)),
            n_fit_after_quality_gate=int(fg_block.get("n_fit_after_quality_gate", 0)),
            kept=int(fg_block.get("kept", 0)),
            dropped=int(fg_block.get("dropped", 0)),
            qualifying_count_distribution=dict(
                fg_block.get("qualifying_count_distribution") or {}
            ),
        )

    intrusion = meta.get("not_fit_intrusion") or {}
    return AnchorResult(
        inputs=inputs,
        calibration=calibration,
        recovery=recovery,
        retrieval=retrieval,
        write=write_result,
        not_fit_intrusions=int(intrusion.get("passed", 0)),
        timings=timings,
        radius_scheme=radius_scheme,
        cohort_consistency=cohort,
        quality_gate_dropped=quality_dropped,
        quality_gate_median_delta_pre_drop=quality_gate_median_pre,
        quality_gate_T_pre_drop=float(quality_block.get("T_pre_drop", 0.0)),
        quality_gate_multiplier=quality_gate_multiplier,
        quality_gate_multiplier_cutoff=quality_gate_multiplier_cutoff,
        # Older sidecars (pre-#47) lack ``median_floor_applied``; default
        # ``False`` keeps replay output for those runs unchanged.
        quality_gate_median_floor_applied=bool(
            quality_block.get("median_floor_applied", False)
        ),
        frequency_gate=frequency_gate,
    )


def hydrate_post_harvest(
    sid: str,
    *,
    user_runs_dir: Path,
    store: "SessionStore",
    user_id: str,
) -> "SessionContext":
    """Rebuild a session frozen at the post-harvest / pre-refine
    boundary.

    Layered on top of :func:`hydrate_post_tuning`: the canonical log
    rebuilds the per-turn :class:`SessionState`, then
    :func:`_reconstruct_anchor_result` rehydrates the Phase 2
    summary so ``/refine/start`` and ``/harvest/result`` see the same
    in-memory shape they would after a fresh harvest run. The session
    lands in ``ANCHOR_DONE`` so the operator's "Continue → Refine"
    click on the SessionList row routes straight into ``/refine/derive``.
    """
    replay = _load_replay(sid, user_runs_dir)
    scope = _read_scope_from_canonical(replay.canonical_path)
    state = _build_state_from_replay(
        replay, scope=scope, user_runs_dir=user_runs_dir
    )
    anchor_result = _reconstruct_anchor_result(replay)
    state.set_phase("ANCHOR_DONE")
    ctx = store.attach(state, user_id=user_id, read_only=False)
    ctx.anchor_result = anchor_result
    return ctx


def _reconstruct_judge_result(replay: ReplaySession, *, rubric_version: int):
    """Rebuild :class:`src.refine.judge.JudgeResult` from
    ``<sid>.phase3.evidence.jsonl``.

    The on-disk record carries every field the dataclass needs except
    ``rubric_version``, which is kept on rubric.json's metadata — we
    pass it in so the rebuilt verdicts agree with the rubric the
    operator is reviewing.
    """
    from src.refine.judge import JudgeResult, JudgeVerdictRecord

    try:
        rows = replay.load_phase3_evidence()
    except ReplayLoadError as exc:
        raise ResumeError(str(exc)) from exc

    verdicts: list[JudgeVerdictRecord] = []
    api_errors = 0
    parse_errors = 0
    total_latency = 0.0
    for row in rows:
        verdict_str = str(row.get("verdict", "ERROR"))
        if verdict_str == "ERROR":
            api_errors += 1
        latency = row.get("latency_ms")
        if isinstance(latency, (int, float)):
            total_latency += float(latency)
        indices_raw = row.get("evidence_line_indices")
        indices: list[int] | None
        if indices_raw is None:
            indices = None
        else:
            indices = [int(i) for i in indices_raw]
        verdicts.append(
            JudgeVerdictRecord(
                pk=row["pk"],
                nearest_fit_distance=float(row.get("nearest_fit_distance") or 0.0),
                decile=int(row.get("decile", 0)),
                chunk_content=str(row.get("chunk_content", "")),
                verdict=verdict_str,
                evidence_line_indices=indices,
                failed_check=row.get("failed_check"),
                reason=str(row.get("reason", "")),
                latency_ms=(
                    float(latency) if isinstance(latency, (int, float)) else None
                ),
                attempts=int(row.get("attempts", 1)),
                rubric_version=int(row.get("rubric_version", rubric_version)),
                prompt_sha256=str(row.get("prompt_sha256", "")),
            )
        )

    return JudgeResult(
        verdicts=verdicts,
        parse_error_count=parse_errors,
        api_error_count=api_errors,
        total_latency_ms=total_latency,
    )


def _rebuild_sample(
    runs_dir: Path,
    *,
    sid: str,
    cfg,
    inputs,
):
    """Re-derive the Phase 3 stratified sample deterministically.

    ``stratified_sample`` is seeded by ``cfg.seed`` and reads the same
    Phase 2 records on disk; calling it again produces an identical
    :class:`StratifiedSample` to the one the live judge run consumed,
    so the rebuilt :class:`RefineRunState` can be threaded into
    ``finalize_refine`` without divergence.
    """
    from src.refine.sample import (
        load_known_intruder_pks,
        load_phase2_records,
        stratified_sample,
    )

    records = load_phase2_records(runs_dir, sid)
    intruders = load_known_intruder_pks(runs_dir, sid)
    return stratified_sample(
        records,
        sample_size=cfg.sample_size,
        n_bins=cfg.n_bins,
        seed=cfg.seed,
        exclude_pks=inputs.rated_pks,
        known_intruder_pks=intruders,
        auto_drop_known_intruders=cfg.auto_drop_known_intruders,
    )


def _build_refine_state(
    replay: ReplaySession,
    *,
    user_runs_dir: Path,
):
    """Reconstruct a :class:`RefineRunState` from Phase 3 sidecars.

    Returns the state with rubric_text + rubric_metadata + judge_result
    + sample populated. ``derive_result`` stays ``None`` (the raw
    derive output is not persisted; the rendered prompt + parsed
    metadata are sufficient for review and finalise) and
    ``write_result`` / ``operator_decision`` are filled in by
    :func:`hydrate_done_view` when phase3.meta.json exists.
    """
    from src.refine.config import load_refine_config
    from src.refine.derive import metadata_from_json, render_rubric_prompt
    from src.refine.errors import RefineConfigError
    from src.refine.load_session import load_phase3_session_inputs
    from src.refine.runner import RefineRunState

    try:
        rubric_payload = replay.load_phase3_rubric()
    except ReplayLoadError as exc:
        raise ResumeError(str(exc)) from exc
    try:
        rubric_metadata = metadata_from_json(json.dumps(rubric_payload))
    except Exception as exc:  # noqa: BLE001 — surface as ResumeError
        raise ResumeError(
            f"Could not parse rubric metadata for {replay.session_id}: {exc}"
        ) from exc

    overrides = _resume_load_overrides(replay.session_id, user_runs_dir)
    try:
        cfg = load_refine_config(session_overrides=overrides.get("refine"))
    except RefineConfigError as exc:
        raise ResumeError(
            f"refine config unavailable for {replay.session_id}: {exc}"
        ) from exc

    try:
        inputs = load_phase3_session_inputs(
            replay.canonical_path, runs_dir=user_runs_dir
        )
    except Exception as exc:  # noqa: BLE001
        raise ResumeError(
            f"Could not load phase 3 inputs for {replay.session_id}: {exc}"
        ) from exc

    judge_result = _reconstruct_judge_result(
        replay, rubric_version=rubric_metadata.version
    )

    try:
        sample = _rebuild_sample(
            user_runs_dir, sid=replay.session_id, cfg=cfg, inputs=inputs
        )
    except Exception as exc:  # noqa: BLE001
        raise ResumeError(
            f"Could not rebuild stratified sample for {replay.session_id}: {exc}"
        ) from exc

    # Prompt.md only lands on disk after stage B (DONE_VIEW). For
    # POST_RUBRIC sessions we re-render it from metadata; the renderer
    # is the inverse of parse_rubric_prompt and is what the live derive
    # path emits, so finalize_refine writes byte-identical output.
    if replay.phase3_prompt_path.exists():
        rubric_text = replay.load_phase3_prompt()
    else:
        rubric_text = render_rubric_prompt(rubric_metadata)

    return RefineRunState(
        inputs=inputs,
        cfg=cfg,
        derive_result=None,
        rubric_text=rubric_text,
        rubric_metadata=rubric_metadata,
        sample=sample,
        judge_result=judge_result,
        write_result=None,
        timings={},
        operator_decision="",
    )


def hydrate_post_rubric(
    sid: str,
    *,
    user_runs_dir: Path,
    store: "SessionStore",
    user_id: str,
) -> "SessionContext":
    """Rebuild a session whose judge run completed and is awaiting
    operator verdict decisions.

    Layers Phase 2 reconstruction (so harvest/refine guards see
    ``anchor_result is not None``) under a freshly rebuilt
    :class:`RefineRunState`. Lands in ``REFINE_REVIEW`` so the
    frontend opens the verdict-review screen with the rubric and
    evidence already loaded; mutating endpoints stay enabled because
    operator decisions and a subsequent finalize are the expected
    next moves.
    """
    replay = _load_replay(sid, user_runs_dir)
    scope = _read_scope_from_canonical(replay.canonical_path)
    state = _build_state_from_replay(
        replay, scope=scope, user_runs_dir=user_runs_dir
    )
    anchor_result = _reconstruct_anchor_result(replay)
    refine_state = _build_refine_state(replay, user_runs_dir=user_runs_dir)
    state.refine_state = refine_state
    state.set_phase("REFINE_REVIEW")
    ctx = store.attach(state, user_id=user_id, read_only=False)
    ctx.anchor_result = anchor_result
    return ctx


def _build_apply_state_view(user_runs_dir: Path, sid: str):
    """Reconstruct a read-only Phase 4 apply state from disk sidecars.

    Returns a duck-typed namespace exposing the same attributes the
    summary serializer and workflow-step builder consume from
    :class:`ApplyRunState` (``classifier_metadata``, ``eval_report``,
    ``write_result``, ``final_projection``, ``operator_decision``).
    ``cohort_p_keep`` is intentionally ``None`` — only the read-side
    callers touch this view, and they only use it to gate "is the
    cohort scored" which is True by definition when ``phase4.meta.json``
    exists.

    Returns ``None`` when any required sidecar is missing — the caller
    treats that as "no apply view available" and leaves
    ``state.apply_state = None``. This is what makes DONE_VIEW resume
    for sessions where Phase 4 never ran fall through to the Phase 3
    summary without raising.
    """
    classifier_path = user_runs_dir / f"{sid}.phase4.classifier.json"
    eval_path = user_runs_dir / f"{sid}.phase4.eval.json"
    meta_path = user_runs_dir / f"{sid}.phase4.meta.json"
    labels_path = user_runs_dir / f"{sid}.phase4.labels.jsonl"

    if not (classifier_path.exists() and eval_path.exists() and meta_path.exists()):
        return None

    from types import SimpleNamespace

    from src.apply.schema import BorderlineSample, CohortProjection, EvalReport
    from src.apply.writer import read_classifier

    try:
        metadata = read_classifier(classifier_path)
        eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
        meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResumeError(f"could not parse phase4 sidecars for {sid}: {exc}") from exc

    # precision_at_threshold / recall_at_threshold must come from the
    # classifier metadata, not from phase4.eval.json. The eval.json
    # sidecar is the Stage A write and stores values at `threshold_default`
    # (the config-default τ). finalize_apply later recomputes those metrics
    # at the operator-chosen τ and overwrites them inside
    # classifier.json.eval_metrics. Reading from eval_payload here surfaced
    # the at-default-τ values on DONE_VIEW resume and made post-calibrate
    # sessions look like a recall regression.
    eval_report = EvalReport(
        precision_at_threshold=metadata.eval_metrics.precision_at_threshold,
        recall_at_threshold=metadata.eval_metrics.recall_at_threshold,
        pr_curve=[tuple(row) for row in eval_payload.get("pr_curve", [])],
        threshold_default=eval_payload["threshold_default"],
        threshold_selected_by_cv=eval_payload.get("threshold_selected_by_cv"),
        cv_precision_mean=eval_payload.get("cv_precision_mean"),
        cv_precision_std=eval_payload.get("cv_precision_std"),
        min_precision=eval_payload["min_precision"],
        eval_n=eval_payload["eval_n"],
        eval_keep_n=eval_payload["eval_keep_n"],
        eval_drop_n=eval_payload["eval_drop_n"],
        borderline_samples=[
            BorderlineSample(
                pk=s["pk"],
                p_keep=s["p_keep"],
                nearest_fit_distance=s["nearest_fit_distance"],
                decile=s["decile"],
            )
            for s in eval_payload.get("borderline_samples", [])
        ],
    )

    cohort = meta_payload.get("cohort_projection") or {}
    final_projection = CohortProjection(
        threshold=cohort.get("threshold", metadata.threshold),
        keep=cohort.get("keep", 0),
        drop=cohort.get("drop", 0),
        total=cohort.get("total", 0),
        per_decile_keep_rate=list(cohort.get("per_decile_keep_rate") or []),
    )

    n_labels = 0
    if labels_path.exists():
        try:
            with labels_path.open("r", encoding="utf-8") as fp:
                n_labels = sum(1 for line in fp if line.strip())
        except OSError:
            n_labels = 0

    write_result = SimpleNamespace(
        classifier_path=classifier_path,
        eval_path=eval_path,
        labels_path=labels_path,
        meta_path=meta_path,
        n_labels=n_labels,
    )

    return SimpleNamespace(
        classifier_metadata=metadata,
        eval_report=eval_report,
        write_result=write_result,
        final_projection=final_projection,
        operator_decision=str(meta_payload.get("operator_decision") or "agree"),
        cohort_p_keep=None,
    )


def hydrate_done_view(
    sid: str,
    *,
    user_runs_dir: Path,
    store: "SessionStore",
    user_id: str,
) -> "SessionContext":
    """Rebuild a finalised session for read-only viewing.

    Builds on POST_RUBRIC's reconstruction and additionally fills in
    :class:`RefineWriteResult` from ``phase3.meta.json`` so
    ``/refine/finalize``-style summary endpoints render the same shape
    they would post-finalize. ``read_only=True`` is forwarded into
    :meth:`SessionStore.attach` so every mutating endpoint 409s — the
    enforcement landed in PR #32 and this hydrator is what arms it.
    """
    from src.refine.writer import RefineWriteResult

    replay = _load_replay(sid, user_runs_dir)
    scope = _read_scope_from_canonical(replay.canonical_path)
    state = _build_state_from_replay(
        replay, scope=scope, user_runs_dir=user_runs_dir
    )
    anchor_result = _reconstruct_anchor_result(replay)
    refine_state = _build_refine_state(replay, user_runs_dir=user_runs_dir)

    try:
        meta = replay.load_phase3_meta()
    except ReplayLoadError as exc:
        raise ResumeError(str(exc)) from exc

    n_verdicts = (
        len(refine_state.judge_result.verdicts)
        if refine_state.judge_result is not None
        else 0
    )
    refine_state.write_result = RefineWriteResult(
        prompt_path=replay.phase3_prompt_path,
        rubric_path=replay.phase3_rubric_path,
        evidence_path=replay.phase3_evidence_path,
        meta_path=replay.phase3_meta_path,
        details_path=replay.details_path,
        n_verdicts=n_verdicts,
    )
    refine_state.operator_decision = str(meta.get("operator_decision") or "agree")

    state.refine_state = refine_state
    # Rehydrate Phase 4 apply state from disk so /apply/summary and the
    # workflow-timeline detail nodes (CALIBRATE τ, APPLY retained) light
    # up on a DONE_VIEW resume. ``None`` when phase4 sidecars are
    # absent — the frontend then falls back to the Phase 3 summary,
    # matching the pre-Phase-4 resume behaviour.
    apply_view = _build_apply_state_view(user_runs_dir, sid)
    if apply_view is not None:
        state.apply_state = apply_view
    state.set_phase("DONE")
    # DONE_VIEW historically froze the session, but Phase 4 (cohort apply)
    # is an additive step that runs *after* refine finalize and writes
    # only phase4.* sidecars. While apply is enabled and labels.jsonl is
    # not on disk, leave the context writable so the operator can train +
    # finalize Phase 4 from a resumed session. Once labels.jsonl exists,
    # the session is fully finalised and reverts to frozen.
    ctx = store.attach(
        state, user_id=user_id, read_only=not _apply_pending(sid, user_runs_dir)
    )
    ctx.anchor_result = anchor_result
    return ctx


def _apply_pending(sid: str, user_runs_dir: Path) -> bool:
    """True iff Phase 4 hasn't shipped labels yet and apply is enabled."""
    labels = user_runs_dir / f"{sid}.phase4.labels.jsonl"
    if labels.exists():
        return False
    try:
        from src.apply.config import load_apply_config

        overrides = _resume_load_overrides(sid, user_runs_dir)
        return bool(load_apply_config(session_overrides=overrides.get("apply")).enabled)
    except Exception:  # noqa: BLE001 — missing/invalid config = treat as off
        return False


def _convergence_from_disk(
    sid: str, user_runs_dir: Path | None
) -> ConvergenceThresholds:
    """Resolve the Phase 1 convergence triple for a resumed session.

    Reads the session's ``<sid>.overrides.json`` sidecar (when
    ``user_runs_dir`` is provided) and merges it onto the current
    ``harvest:`` YAML block. Falls back to the module defaults when
    the sidecar is missing, the YAML is malformed, or no runs dir
    was supplied (legacy in-process callers / tests).
    """
    if user_runs_dir is None:
        return ConvergenceThresholds.from_module_defaults()
    overrides = _resume_load_overrides(sid, user_runs_dir)
    harvest_block = overrides.get("harvest")
    try:
        cfg = load_harvest_config(session_overrides=harvest_block)
    except AnchorError as exc:
        log.warning(
            "Resume: falling back to module convergence defaults for %s; "
            "harvest config failed to load: %s",
            sid,
            exc,
        )
        return ConvergenceThresholds.from_module_defaults()
    return ConvergenceThresholds(
        min_fit=cfg.min_fit,
        min_not_fit=cfg.min_not_fit,
        precision_at_k=cfg.precision_at_k,
    )


def _resume_load_overrides(sid: str, user_runs_dir: Path) -> dict[str, dict[str, Any]]:
    """Load ``<sid>.overrides.json`` from disk for the resume path.

    Mirrors the worker-side helper in :mod:`src.web_api.app` so resume-
    time config decisions (enable/disable Phase 3 or 4, threshold pre-
    fill) see the same overrides as live Phase runs.
    """
    from src.config_loader import ConfigFileError, load_session_overrides

    try:
        return load_session_overrides(sid, user_runs_dir, user_id=None)
    except ConfigFileError:
        return {}


# Abstract steps that the replay advance endpoint walks the user
# through. Distinct from the live ``state.phase`` enum because the
# replay UX collapses the interactive review screens (REFINE_REVIEW,
# APPLY_REVIEW) into their post-finalize summary views.
REPLAY_STEP_TUNING = "TUNING"
REPLAY_STEP_HARVEST = "REPLAY_HARVEST"
REPLAY_STEP_REFINE = "REPLAY_REFINE"
REPLAY_STEP_APPLY = "REPLAY_APPLY"
REPLAY_STEP_DONE = "REPLAY_DONE"


def hydrate_replay(
    sid: str,
    *,
    user_runs_dir: Path,
    store: "SessionStore",
    user_id: str,
) -> "SessionContext":
    """Hydrate a past session for read-only Replay Mode.

    Unlike the regular resume hydrators, replay **always lands at
    Phase 1, turn 1** regardless of how far the original session
    progressed. The phase-2/3/4 reconstructions are loaded eagerly into
    the context but kept dormant on ``ctx.anchor_result``/``state.refine_state``/
    ``state.apply_state``; the replay advance endpoint rotates them in
    one at a time.

    Sets ``read_only=True`` on the context so the existing
    ``_require_writable`` guard rejects every mutating endpoint.
    """
    replay = _load_replay(sid, user_runs_dir)
    if not replay.turns:
        raise ResumeError(f"Session {sid} has no Phase 1 turn rows; nothing to replay")
    scope = _read_scope_from_canonical(replay.canonical_path)

    # Pre-build phase-2/3/4 views from whatever sidecars exist. We
    # tolerate missing later-phase sidecars (a session may have ended
    # at Phase 1) but a half-written phase-3/4 set is a hard error so
    # the user finds out before they start the replay.
    phase_sequence: list[str] = [REPLAY_STEP_TUNING]
    anchor_result = None
    refine_state = None
    apply_view = None

    if replay.phase2_meta_path.exists():
        anchor_result = _reconstruct_anchor_result(replay)
        phase_sequence.append(REPLAY_STEP_HARVEST)

    if replay.phase3_meta_path.exists():
        refine_state = _build_refine_state(replay, user_runs_dir=user_runs_dir)
        # Mirror hydrate_done_view: attach a WriteResult so the refine
        # summary endpoint serves the same shape it would post-finalize.
        from src.refine.writer import RefineWriteResult

        try:
            meta = replay.load_phase3_meta()
        except ReplayLoadError as exc:
            raise ResumeError(str(exc)) from exc
        n_verdicts = (
            len(refine_state.judge_result.verdicts)
            if refine_state.judge_result is not None
            else 0
        )
        refine_state.write_result = RefineWriteResult(
            prompt_path=replay.phase3_prompt_path,
            rubric_path=replay.phase3_rubric_path,
            evidence_path=replay.phase3_evidence_path,
            meta_path=replay.phase3_meta_path,
            details_path=replay.details_path,
            n_verdicts=n_verdicts,
        )
        refine_state.operator_decision = str(meta.get("operator_decision") or "agree")
        phase_sequence.append(REPLAY_STEP_REFINE)

    phase4_meta = user_runs_dir / f"{sid}.phase4.meta.json"
    if phase4_meta.exists():
        apply_view = _build_apply_state_view(user_runs_dir, sid)
        if apply_view is not None:
            phase_sequence.append(REPLAY_STEP_APPLY)

    phase_sequence.append(REPLAY_STEP_DONE)

    # Build a fresh SessionState parked at the first Phase 1 turn.
    # ``state.turns`` is empty — we want the UI to render turn 1 as
    # the *current* turn, not a historical one — and ``current_table``
    # holds the historical evidence with its original ratings.
    first_turn = replay.turns[0]
    state = SessionState(
        current_config=first_turn.config,
        session_id=replay.session_id,
    )
    state.scope = scope
    _rehydrate_tuning_at_cursor(state, list(replay.turns), 0)
    state.set_phase("TUNING")

    ctx = store.attach(state, user_id=user_id, read_only=True, replay=True)
    ctx.replay_turns = list(replay.turns)
    ctx.replay_cursor = 0
    ctx.replay_phase_sequence = phase_sequence
    ctx.replay_phase_index = 0
    ctx.replay_anchor_result = anchor_result
    ctx.replay_refine_state = refine_state
    ctx.replay_apply_view = apply_view
    return ctx


def advance_replay(ctx: "SessionContext") -> None:
    """Mutate ``ctx`` in place to the next step of the replay.

    Idempotent at the terminal step. The state machine is documented
    inline; see ``hydrate_replay`` for the abstract step labels.
    """
    state = ctx.state
    # First, try advancing within the current step.
    current_step = ctx.replay_phase_sequence[ctx.replay_phase_index]
    if current_step == REPLAY_STEP_TUNING:
        next_cursor = ctx.replay_cursor + 1
        if next_cursor < len(ctx.replay_turns):
            ctx.replay_cursor = next_cursor
            _rehydrate_tuning_at_cursor(state, ctx.replay_turns, next_cursor)
            return
        # End of phase 1 — fall through to next-step transition.

    # Step boundary — advance the phase index, but never past the end.
    if ctx.replay_phase_index + 1 >= len(ctx.replay_phase_sequence):
        return
    ctx.replay_phase_index += 1
    next_step = ctx.replay_phase_sequence[ctx.replay_phase_index]

    if next_step == REPLAY_STEP_HARVEST:
        # Backfill state.turns with the full phase-1 history so the
        # workflow timeline + convergence panel reflect the original
        # session, then surface the harvest summary.
        _backfill_state_turns(state, ctx.replay_turns)
        ctx.anchor_result = ctx.replay_anchor_result
        state.set_phase("ANCHOR_DONE")
        return
    if next_step == REPLAY_STEP_REFINE:
        # Use DONE phase + only refine_state populated so the existing
        # Rating screen renders RefineSummary (the post-finalize view).
        state.refine_state = ctx.replay_refine_state
        state.apply_state = None
        state.set_phase("DONE")
        return
    if next_step == REPLAY_STEP_APPLY:
        # DONE phase + apply_state populated -> ApplyDoneSummary renders.
        state.apply_state = ctx.replay_apply_view
        state.set_phase("DONE")
        return
    if next_step == REPLAY_STEP_DONE:
        # Terminal — every prior step's state remains attached. No
        # transitions left; subsequent advance calls are idempotent.
        state.set_phase("DONE")
        return


def _rehydrate_tuning_at_cursor(
    state: SessionState, replay_turns: list, cursor: int
) -> None:
    """Snap ``state`` to look like the operator is mid-rating turn
    ``cursor + 1`` of a fresh session.

    ``state.turns`` is rebuilt as the history *before* the current
    cursor so ``state.turn_number`` reads ``cursor + 1`` — the same
    value the HeaderBar would show on a live session at this turn.
    ``current_table`` carries the historical rated evidence so the
    Rating screen renders the original chunks with their original
    ratings intact (visually identical to the operator's first run).
    """
    state.turns.clear()
    state.seen_pks.clear()
    for prior in replay_turns[:cursor]:
        state.turns.append(
            TurnRecord(
                turn_number=prior.turn_number,
                query=prior.query,
                config=prior.config,
                evidence_table=prior.evidence_table,
                breakdown=prior.breakdown,
                precision=prior.precision,
                reflection=prior.reflection,
                audit_turn=False,
            )
        )
        for row in prior.evidence_table.rows:
            state.seen_pks.add(row.pk)
        for candidates in prior.evidence_table.per_path_candidates.values():
            for cand in candidates:
                state.seen_pks.add(cand.pk)
    current = replay_turns[cursor]
    state.current_table = current.evidence_table
    state.current_config = current.config
    state.query = current.query


def _backfill_state_turns(state: SessionState, replay_turns: list) -> None:
    """Populate ``state.turns`` with the full phase-1 history.

    Used at the TUNING → HARVEST transition so downstream consumers
    (workflow timeline, convergence DTO, breakdown_by_turn) reflect
    the same shape they would on a normally-completed session.
    """
    if state.turns:
        return  # already backfilled (idempotent advance)
    for replay_turn in replay_turns:
        record = TurnRecord(
            turn_number=replay_turn.turn_number,
            query=replay_turn.query,
            config=replay_turn.config,
            evidence_table=replay_turn.evidence_table,
            breakdown=replay_turn.breakdown,
            precision=replay_turn.precision,
            reflection=replay_turn.reflection,
            audit_turn=False,
        )
        state.turns.append(record)
        for row in replay_turn.evidence_table.rows:
            state.seen_pks.add(row.pk)
        for candidates in replay_turn.evidence_table.per_path_candidates.values():
            for cand in candidates:
                state.seen_pks.add(cand.pk)
    # current_table is cleared so the snapshot serializer falls back to
    # the last completed turn (the existing "converged" rendering path).
    state.current_table = None


__all__ = [
    "REPLAY_STEP_APPLY",
    "REPLAY_STEP_DONE",
    "REPLAY_STEP_HARVEST",
    "REPLAY_STEP_REFINE",
    "REPLAY_STEP_TUNING",
    "ResumeError",
    "ResumeNotImplementedError",
    "ResumeTarget",
    "SessionListEntry",
    "advance_replay",
    "classify",
    "hydrate_done_view",
    "hydrate_post_harvest",
    "hydrate_post_rubric",
    "hydrate_post_tuning",
    "hydrate_replay",
    "list_user_sessions",
]
