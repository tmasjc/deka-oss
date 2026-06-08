"""Convert internal dataclasses to API DTOs.

Kept in a dedicated module so :mod:`app` stays focused on HTTP wiring
and the TUI's state objects stay framework-agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.search.config import SearchConfig
from src.search.evidence import (
    ALL_PATHS,
    CandidateRow,
    EvidenceRow,
    EvidenceTable,
    compute_breakdown,
)
from src.session.state import SessionState

from .schemas import (
    AnchorResultDTO,
    ApplyEvalReportDTO,
    ApplySummaryDTO,
    BorderlineSampleDTO,
    BreakdownRowDTO,
    CandidateRowDTO,
    CohortMissingDTO,
    CohortProjectionDTO,
    ConvergenceDTO,
    DecileRowDTO,
    DeriveResultDTO,
    EvidenceRowDTO,
    EvidenceTableDTO,
    FrequencyGateDTO,
    HarvestTimingsDTO,
    JudgeDecileBucketDTO,
    JudgeResultDTO,
    PRCurvePointDTO,
    ParamsDTO,
    PathDropRecommendationDTO,
    ProbeSummaryDTO,
    QualityGateDropDTO,
    ReflectionDTO,
    RefineSummaryDTO,
    RubricCheckDTO,
    RubricExampleDTO,
    RubricMetadataDTO,
    SessionSnapshot,
    TurnBreakdownDTO,
    VerdictDTO,
    WorkflowStepDTO,
)


def row_to_dto(row: EvidenceRow) -> EvidenceRowDTO:
    return EvidenceRowDTO(
        rank=row.rank,
        pk=row.pk,
        chunk_id=row.chunk_id,
        chunk_content=row.chunk_content,
        sample_id=row.sample_id,
        counselor_id=row.counselor_id,
        term=row.term,
        source_paths=list(row.source_paths),
        scores=dict(row.scores),
        rating=row.rating,
        span_line_indices=list(row.span_line_indices),
        span_text=row.span_text,
    )


def candidate_to_dto(cand: CandidateRow) -> CandidateRowDTO:
    return CandidateRowDTO(
        path=cand.path,
        rank_in_path=cand.rank_in_path,
        pk=cand.pk,
        chunk_id=cand.chunk_id,
        chunk_content=cand.chunk_content,
        sample_id=cand.sample_id,
        counselor_id=cand.counselor_id,
        term=cand.term,
        score=cand.score,
        rating=cand.rating,
        span_line_indices=list(cand.span_line_indices),
        span_text=cand.span_text,
    )


def table_to_dto(table: EvidenceTable) -> EvidenceTableDTO:
    return EvidenceTableDTO(
        query=table.query,
        rows=[row_to_dto(r) for r in table.rows],
        per_path_candidates={
            path: [candidate_to_dto(c) for c in table.per_path_candidates.get(path, [])]
            for path in ALL_PATHS
        },
        filtered_short_chunk=table.filtered_short_chunk,
        filtered_duplicate_sample=table.filtered_duplicate_sample,
        dropped_by_extractor=table.dropped_by_extractor,
    )


def config_to_params(config: SearchConfig) -> ParamsDTO:
    return ParamsDTO(
        rrf_k=config.rrf_k,
        per_path_limit=config.per_path_limit,
        top_k=config.top_k,
        active_paths=sorted(config.active_paths),
    )


def convergence_dto(state: SessionState) -> ConvergenceDTO:
    pk_current = state.turns[-1].precision if state.turns else 0.0
    fit_current = len(state.cumulative_fit_pks)
    not_fit_current = len(state.cumulative_not_fit_pks)
    thresholds = state.convergence
    return ConvergenceDTO(
        pk_current=round(pk_current, 4),
        fit_current=fit_current,
        not_fit_current=not_fit_current,
        pk_threshold=thresholds.precision_at_k,
        fit_threshold=thresholds.min_fit,
        not_fit_threshold=thresholds.min_not_fit,
        converged=state.is_converged,
    )


def workflow_steps(
    state: SessionState, *, anchor_result: Any = None
) -> list[WorkflowStepDTO]:
    """Render the workflow timeline shown in the right panel.

    One node per completed turn with its P@K as detail, then a pending
    node for the in-progress turn while still TUNING, then CONVERGED
    and HARVEST as terminal nodes.
    """
    converged = state.is_converged
    harvest_active = state.phase.startswith("ANCHOR_")
    harvest_done = state.phase == "ANCHOR_DONE" and anchor_result is not None
    harvest_failed = state.phase == "ANCHOR_FAILED"

    def step(
        key: str, label: str, status: str, detail: str | None = None
    ) -> WorkflowStepDTO:
        return WorkflowStepDTO(key=key, status=status, label=label, detail=detail)  # type: ignore[arg-type]

    steps: list[WorkflowStepDTO] = [step("START", "START", "done")]

    for i, turn in enumerate(state.turns, start=1):
        steps.append(
            step(f"TURN_{i}", f"TURN {i}", "done", f"P@K {turn.precision:.2f}")
        )

    # Surface the in-progress turn as a pending step while we're still
    # in the tuning loop. Gate on phase, not on ``is_converged`` — the
    # operator-override-low-precision path advances past TUNING without
    # the metric predicate ever flipping True, and a phase-past-TUNING
    # session must not render a phantom pending turn alongside the
    # already-done CONVERGED / HARVEST nodes.
    if state.phase == "TUNING":
        pending_n = len(state.turns) + 1
        steps.append(
            step(f"TURN_{pending_n}", f"TURN {pending_n}", "current", "pending")
        )

    refine_phase = state.phase if state.phase.startswith("REFINE_") else None
    refine_done = state.phase == "DONE"
    apply_phase = state.phase if state.phase.startswith("APPLY_") else None
    apply_visible = apply_phase is not None or (
        refine_done and getattr(state, "apply_state", None) is not None
    )

    if harvest_done or refine_phase or refine_done or apply_phase:
        retained = (
            anchor_result.write.n_records
            if anchor_result and anchor_result.write is not None
            else (
                len(anchor_result.retrieval.candidates)
                if anchor_result is not None
                else 0
            )
        )
        steps.append(step("CONVERGED", "CONVERGED", "done"))
        steps.append(step("HARVEST", "HARVEST", "done", f"{retained} retained"))
        rubric_status, rubric_detail = _refine_node_state(state.phase, "rubric")
        sample_status, sample_detail = _refine_node_state(state.phase, "sample")
        # When the upstream node-state mapper has marked a step "done",
        # replace its generic detail (None) with a concrete artefact
        # value pulled from refine_state / apply_state. The mapper itself
        # stays state-machine-only (no knowledge of refine artefacts) so
        # the in-progress phases keep their "deriving" / "judging" /
        # "calibrating" hints.
        refine_st = getattr(state, "refine_state", None)
        if rubric_status == "done" and refine_st is not None:
            rmeta = getattr(refine_st, "rubric_metadata", None)
            if rmeta is not None and getattr(rmeta, "version", None) is not None:
                rubric_detail = f"v{rmeta.version}"
        if sample_status == "done" and refine_st is not None:
            cfg = getattr(refine_st, "cfg", None)
            if cfg is not None and getattr(cfg, "sample_size", None) is not None:
                sample_detail = f"{cfg.sample_size}"
        steps.append(step("RUBRIC", "RUBRIC", rubric_status, rubric_detail))
        steps.append(step("SAMPLE", "SAMPLE", sample_status, sample_detail))
        if apply_visible:
            apply_st = getattr(state, "apply_state", None)
            ap_meta = (
                getattr(apply_st, "classifier_metadata", None) if apply_st else None
            )
            ap_proj = getattr(apply_st, "final_projection", None) if apply_st else None
            for key, label, status, detail in _apply_substages(state):
                if status == "done":
                    if key == "CALIBRATE" and ap_meta is not None:
                        detail = f"τ={ap_meta.threshold:.2f}"
                    elif key == "APPLY" and ap_proj is not None:
                        detail = f"{ap_proj.keep} retained"
                steps.append(step(key, label, status, detail))
    elif harvest_failed:
        steps.append(step("CONVERGED", "CONVERGED", "done"))
        steps.append(step("HARVEST", "HARVEST", "current", "failed"))
    elif harvest_active:
        detail = state.phase.replace("ANCHOR_", "").lower()
        steps.append(step("CONVERGED", "CONVERGED", "done"))
        steps.append(step("HARVEST", "HARVEST", "current", detail))
    elif converged:
        steps.append(step("CONVERGED", "CONVERGED", "current"))
        steps.append(step("HARVEST", "HARVEST", "pending"))
    else:
        steps.append(step("CONVERGED", "CONVERGED", "pending"))
        steps.append(step("HARVEST", "HARVEST", "pending"))

    return steps


def _refine_node_state(phase: str, which: str) -> tuple[str, str | None]:
    """Map the SessionState phase to (status, detail) for the RUBRIC
    and SAMPLE timeline nodes.
    """
    if which == "rubric":
        if phase in ("REFINE_DERIVING", "REFINE_EDITING"):
            return ("current", phase.replace("REFINE_", "").lower())
        if phase in (
            "REFINE_JUDGING",
            "REFINE_REVIEW",
            "REFINE_FAILED",
            "DONE",
        ) or phase.startswith("APPLY_"):
            return ("done", None)
        return ("pending", None)
    # sample
    if phase == "REFINE_JUDGING":
        return ("current", "judging")
    if phase == "REFINE_REVIEW":
        return ("current", "review")
    if phase == "DONE" or phase.startswith("APPLY_"):
        return ("done", None)
    if phase == "REFINE_FAILED":
        return ("current", "failed")
    return ("pending", None)


def _apply_substages(
    state: SessionState,
) -> list[tuple[str, str, str, str | None]]:
    """Render Phase 4 as TRAIN → CALIBRATE → SHIP substages.

    Returns a list of ``(key, label, status, detail)`` tuples in
    timeline order. The phase determines which substage is current;
    APPLY_FAILED is bucketed by inspecting ``apply_state`` to decide
    whether the failure happened before the cohort was scored (treated
    as TRAIN failure) or after (SHIP failure). The KEY stays
    ``"APPLY"`` since it's the stable internal id (test fixtures pin
    to it); only the LABEL changes to the user-visible verb.
    """
    phase = state.phase
    ap = getattr(state, "apply_state", None)
    has_cohort = ap is not None and getattr(ap, "cohort_p_keep", None) is not None

    train_s, train_d = "pending", None
    cal_s, cal_d = "pending", None
    apl_s, apl_d = "pending", None

    if phase == "APPLY_TRAINING":
        train_s, train_d = "current", "training"
    elif phase == "APPLY_PREPARING":
        train_s, train_d = "current", "preparing"
    elif phase == "APPLY_REVIEW":
        train_s = "done"
        cal_s, cal_d = "current", "calibrating"
    elif phase == "APPLY_CONFIRM":
        train_s = "done"
        cal_s = "done"
        apl_s, apl_d = "current", "confirm"
    elif phase == "APPLY_APPLYING":
        train_s = "done"
        cal_s = "done"
        apl_s, apl_d = "current", "writing"
    elif phase == "DONE" and ap is not None:
        train_s = "done"
        cal_s = "done"
        apl_s = "done"
    elif phase == "APPLY_FAILED":
        if not has_cohort:
            train_s, train_d = "current", "failed"
        else:
            train_s = "done"
            cal_s = "done"
            apl_s, apl_d = "current", "failed"

    return [
        ("TRAIN", "TRAIN", train_s, train_d),
        ("CALIBRATE", "CALIBRATE", cal_s, cal_d),
        ("APPLY", "SHIP", apl_s, apl_d),
    ]


_BREAKDOWN_KEYS = ("dense_only", "sparse_only", "multi_path")


def _breakdown_to_dto(
    per_turn: dict[str, dict[str, int]],
) -> dict[str, BreakdownRowDTO]:
    out: dict[str, BreakdownRowDTO] = {}
    for key in _BREAKDOWN_KEYS:
        row = per_turn.get(key, {})
        out[key] = BreakdownRowDTO(
            total=row.get("total", 0),
            fit=row.get("fit", 0),
            not_fit=row.get("not_fit", 0),
            discard=row.get("discard", 0),
        )
    return out


def breakdown_by_turn(state: SessionState) -> list[TurnBreakdownDTO]:
    """Per-turn FIT/NOT_FIT counts keyed by which path(s) returned the row."""
    out: list[TurnBreakdownDTO] = []
    for turn in state.turns:
        try:
            per_turn = turn.breakdown or compute_breakdown(turn.evidence_table)
        except ValueError:
            continue
        out.append(
            TurnBreakdownDTO(
                turn=turn.turn_number,
                breakdown=_breakdown_to_dto(per_turn),
            )
        )
    return out


def drop_impact_preview(state: SessionState) -> dict[str, Any] | None:
    """Pull ``drop_previews`` off the active table's search_diagnostics."""
    table = state.current_table
    if table is None or not table.search_diagnostics:
        return None
    previews = table.search_diagnostics.get("drop_previews")
    if not previews:
        return None
    return dict(previews)


def cumulative_breakdown(state: SessionState) -> dict[str, BreakdownRowDTO]:
    """Sum per-path breakdowns across all completed turns."""
    totals: dict[str, dict[str, int]] = {
        "dense_only": {"total": 0, "fit": 0, "not_fit": 0, "discard": 0},
        "sparse_only": {"total": 0, "fit": 0, "not_fit": 0, "discard": 0},
        "multi_path": {"total": 0, "fit": 0, "not_fit": 0, "discard": 0},
    }
    for turn in state.turns:
        try:
            per_turn = turn.breakdown or compute_breakdown(turn.evidence_table)
        except ValueError:
            continue
        for key, row in per_turn.items():
            bucket = totals.setdefault(
                key, {"total": 0, "fit": 0, "not_fit": 0, "discard": 0}
            )
            for field in ("total", "fit", "not_fit", "discard"):
                bucket[field] += row.get(field, 0)
    return {k: BreakdownRowDTO(**v) for k, v in totals.items()}


def snapshot(
    state: SessionState,
    *,
    anchor_result: Any = None,
    read_only: bool = False,
    replay: bool = False,
) -> SessionSnapshot:
    table = state.current_table
    turn_complete = state.all_rated()
    if table is None:
        # Convergence path: complete_turn() cleared current_table and the
        # orchestrator returned early without start_turn(). Fall back to the
        # last completed turn so the UI can render the terminal state.
        if state.is_converged and state.turns:
            table = state.turns[-1].evidence_table
            turn_complete = True
        else:
            raise RuntimeError("Cannot snapshot a session with no active turn")
    probe_summary: ProbeSummaryDTO | None = None
    if state.probe_summary is not None:
        probe_summary = ProbeSummaryDTO.model_validate(state.probe_summary)
    return SessionSnapshot(
        session_id=state.session_id,
        query=state.query or table.query,
        turn_number=state.turn_number,
        phase=state.phase,
        scope=state.scope or "",
        table=table_to_dto(table),
        params=config_to_params(state.current_config),
        convergence=convergence_dto(state),
        workflow=workflow_steps(state, anchor_result=anchor_result),
        breakdown_cumulative=cumulative_breakdown(state),
        precision_trend=list(state.precision_trend),
        breakdown_by_turn=breakdown_by_turn(state),
        drop_impact_preview=drop_impact_preview(state),
        turn_complete=turn_complete,
        audit_mode_active=state.audit_mode_active,
        probe_summary=probe_summary,
        read_only=read_only,
        replay=replay,
    )


def anchor_result_to_dto(result: Any) -> AnchorResultDTO:
    """Convert :class:`src.anchor.runner.AnchorResult` to the DTO shown
    in the harvest summary screen.
    """
    from src.anchor.threshold import distance_summary

    calib = result.calibration
    delta_stats = distance_summary(calib.deltas)
    t_prime_stats = distance_summary(calib.T_primes)

    retained = (
        result.write.n_records
        if result.write is not None
        else len(result.retrieval.candidates)
    )

    freq_dto: FrequencyGateDTO | None = None
    fg = result.frequency_gate
    if fg is not None:
        freq_dto = FrequencyGateDTO(
            f_configured=fg.f_configured,
            n_fit_after_quality_gate=fg.n_fit_after_quality_gate,
            kept=fg.kept,
            dropped=fg.dropped,
            qualifying_count_distribution=dict(fg.qualifying_count_distribution),
            qualifying_count_histogram={
                str(k): int(v) for k, v in fg.qualifying_count_histogram.items()
            },
        )

    quality_drops = [
        QualityGateDropDTO(
            fit_chunk_id=d["fit_chunk_id"],
            delta=float(d["delta"]),
            reasons=list(d["reasons"]),
        )
        for d in result.quality_gate_dropped
    ]
    cohort_missing = [
        CohortMissingDTO(fit_chunk_id=rec["fit_chunk_id"])
        for rec in result.cohort_consistency
        if not rec["own_chunk_retained"]
    ]
    budget_exhausted = [
        page.fit_chunk_id
        for page in result.retrieval.per_fit_pages
        if page.budget_exhausted
    ]

    sidecar_jsonl: str | None = None
    sidecar_meta: str | None = None
    if result.write is not None:
        sidecar_jsonl = str(result.write.jsonl_path)
        sidecar_meta = str(result.write.meta_path)

    # ``n_fit_entering_quality_gate`` is the cohort size the gate
    # inspected — dropped + survivors. The survivor count is
    # ``frequency_gate.n_fit_after_quality_gate`` when the frequency
    # gate ran, otherwise the post-LOO total (dry-run path).
    if freq_dto is not None:
        survivors = freq_dto.n_fit_after_quality_gate
    else:
        survivors = result.recovery.total
    n_entering = len(quality_drops) + survivors

    multiplier_cutoff = result.quality_gate_multiplier_cutoff
    return AnchorResultDTO(
        verdict=result.recovery.verdict,
        loo_recovered=result.recovery.recovered,
        loo_total=result.recovery.total,
        T=float(calib.T),
        delta_min=float(delta_stats["min"]),
        delta_median=float(delta_stats["median"]),
        delta_max=float(delta_stats["max"]),
        T_prime_min=float(t_prime_stats["min"]),
        T_prime_median=float(t_prime_stats["median"]),
        T_prime_max=float(t_prime_stats["max"]),
        T_prime_out=float(calib.T_prime_out),
        radius_scheme=result.radius_scheme.value,
        retained_chunks=retained,
        not_fit_intrusions=result.not_fit_intrusions,
        quality_gate_median_delta_pre_drop=float(
            result.quality_gate_median_delta_pre_drop
        ),
        quality_gate_T_pre_drop=float(result.quality_gate_T_pre_drop),
        quality_gate_multiplier=float(result.quality_gate_multiplier),
        quality_gate_multiplier_cutoff=(
            float(multiplier_cutoff) if multiplier_cutoff is not None else None
        ),
        quality_gate_median_floor_applied=bool(
            result.quality_gate_median_floor_applied
        ),
        n_fit_entering_quality_gate=int(n_entering),
        n_discard_filtered=int(result.n_discard_filtered),
        frequency_gate=freq_dto,
        quality_gate_dropped=quality_drops,
        cohort_consistency_missing=cohort_missing,
        budget_exhausted=budget_exhausted,
        sidecar_jsonl_path=sidecar_jsonl,
        sidecar_meta_path=sidecar_meta,
        timings=HarvestTimingsDTO(
            load_ms=result.timings.load_ms,
            calibrate_ms=result.timings.calibrate_ms,
            loo_ms=result.timings.loo_ms,
            retrieve_ms=result.timings.retrieve_ms,
            total_ms=result.timings.total_ms,
        ),
    )


def rubric_metadata_to_dto(meta: Any) -> RubricMetadataDTO:
    """Convert :class:`src.refine.schema.RubricMetadata` to a DTO.

    Strips the SHA / path fields that are internal to the persistence
    layer; the editor only needs the operator-facing structure.
    """
    return RubricMetadataDTO(
        query=meta.query,
        derive_model_id=meta.derive_model_id,
        checks=[
            RubricCheckDTO(id=c.id, description=c.description, required=c.required)
            for c in meta.checks
        ],
        fit_examples=[
            RubricExampleDTO(pk=ex.pk, span_text=ex.span_text)
            for ex in meta.fit_examples
        ],
        not_fit_examples=[
            RubricExampleDTO(pk=ex.pk, span_text=ex.span_text, fails=list(ex.fails))
            for ex in meta.not_fit_examples
        ],
        version=meta.version,
    )


def derive_result_to_dto(derive: Any, meta: Any) -> DeriveResultDTO:
    return DeriveResultDTO(
        rubric_text=derive.rubric_text,
        metadata=rubric_metadata_to_dto(meta),
        attempts=derive.attempts,
        latency_ms=derive.latency_ms,
    )


def judge_result_to_dto(jr: Any) -> JudgeResultDTO:
    """Aggregate per-decile counts from the verdicts list.

    The runner doesn't pre-compute decile breakdowns — we do it here
    so the shape matches the TUI's :class:`RefineSummaryScreen` table.
    """
    keep = drop = err = 0
    deciles: dict[int, dict[str, int]] = {}
    for v in jr.verdicts:
        if v.verdict == "KEEP":
            keep += 1
        elif v.verdict == "DROP":
            drop += 1
        else:
            err += 1
        bucket = deciles.setdefault(
            v.decile, {"n": 0, "keep": 0, "drop": 0, "error": 0}
        )
        bucket["n"] += 1
        bucket[v.verdict.lower() if v.verdict in ("KEEP", "DROP") else "error"] += 1

    decile_dto = {str(k): JudgeDecileBucketDTO(**v) for k, v in sorted(deciles.items())}
    return JudgeResultDTO(
        keep_count=keep,
        drop_count=drop,
        error_count=err,
        parse_error_count=jr.parse_error_count,
        api_error_count=jr.api_error_count,
        total_latency_ms=jr.total_latency_ms,
        decile_breakdown=decile_dto,
    )


def verdicts_to_dto_list(verdicts: list[Any]) -> list[VerdictDTO]:
    """Serialise judge verdicts. ``evidence_line_indices`` are stored
    1-based per the rubric judge contract (see
    :mod:`src.refine.schema`); we normalise to 0-based here so the
    frontend's single ``spanLines`` convention (Phase 1's
    ``span_line_indices`` is already 0-based) lights up the right
    rows in ``ChunkCard``.
    """
    out: list[VerdictDTO] = []
    for v in verdicts:
        raw = v.evidence_line_indices or []
        zero_based = [i - 1 for i in raw if isinstance(i, int) and i >= 1]
        out.append(
            VerdictDTO(
                pk=v.pk,
                nearest_fit_distance=v.nearest_fit_distance,
                decile=v.decile,
                chunk_content=v.chunk_content,
                verdict=v.verdict,  # type: ignore[arg-type]
                evidence_line_indices=zero_based,
                failed_check=v.failed_check,
                reason=v.reason,
            )
        )
    return out


def refine_summary_to_dto(rs: Any) -> RefineSummaryDTO:
    """Compose the terminal Phase 3 summary payload.

    Verdict counts split out auto-drops, rubric version in the header,
    a per-decile table with distance ranges + sample keep rate, and an
    "Estimated Total Chunks" projection (Σ keep_rate × population per
    bin) computed from the sample's ``per_decile_count`` × the sample-
    level keep rate.
    """
    jr = rs.judge_result
    sample = rs.sample
    n_bins = rs.cfg.n_bins if rs.cfg is not None else 10

    # Verdict counts — auto-drops are the auto_drop_known_intruder
    # rows the runner stamps with verdict="DROP" but a sentinel
    # failed_check. Pull them out so DROP only carries judge-decided
    # drops.
    keep = drop = err = auto = 0
    for v in jr.verdicts:
        if v.verdict == "ERROR":
            err += 1
        elif v.failed_check == "auto_drop_known_intruder":
            auto += 1
        elif v.verdict == "KEEP":
            keep += 1
        elif v.verdict == "DROP":
            drop += 1

    # Per-decile keep rate — same predicate as
    # ``src.refine.writer._per_decile_keep_rate``: skip ERROR + auto.
    bin_keep = [0] * n_bins
    bin_drop = [0] * n_bins
    bin_dist_min: list[float | None] = [None] * n_bins
    bin_dist_max: list[float | None] = [None] * n_bins
    for v in jr.verdicts:
        if v.verdict == "ERROR":
            continue
        if v.failed_check == "auto_drop_known_intruder":
            continue
        d = v.decile
        if not 0 <= d < n_bins:
            continue
        if v.verdict == "KEEP":
            bin_keep[d] += 1
        elif v.verdict == "DROP":
            bin_drop[d] += 1
        dist = v.nearest_fit_distance
        if bin_dist_min[d] is None or dist < bin_dist_min[d]:  # type: ignore[operator]
            bin_dist_min[d] = dist
        if bin_dist_max[d] is None or dist > bin_dist_max[d]:  # type: ignore[operator]
            bin_dist_max[d] = dist

    # Decile boundaries from the sampler if available (preferred —
    # they're the bin edges, not just observed extrema). Fall back to
    # observed min/max from the verdicts when sample is None
    # (shouldn't happen post-judge, but the DTO must be safe).
    boundaries: list[float] | None = None
    population_count: list[int] | None = None
    if sample is not None:
        boundaries = list(sample.decile_boundaries)
        population_count = list(sample.per_decile_count)

    rows: list[DecileRowDTO] = []
    estimated_total = 0
    for i in range(n_bins):
        if boundaries is not None and i + 1 < len(boundaries):
            lo: float | None = boundaries[i]
            hi: float | None = boundaries[i + 1]
        else:
            lo = bin_dist_min[i]
            hi = bin_dist_max[i]

        n = (
            population_count[i]
            if population_count is not None and i < len(population_count)
            else 0
        )
        decided = bin_keep[i] + bin_drop[i]
        rate: float | None = (bin_keep[i] / decided) if decided > 0 else None
        if rate is None:
            k_proj = 0
            d_proj = 0
        else:
            k_proj = round(rate * n)
            d_proj = n - k_proj
        estimated_total += k_proj
        rows.append(
            DecileRowDTO(
                decile=i + 1,
                distance_min=lo,
                distance_max=hi,
                sample_n=n,
                keep_count=k_proj,
                drop_count=d_proj,
                keep_rate=rate,
            )
        )

    sidecars: dict[str, str] = {}
    if rs.write_result is not None:
        wr = rs.write_result
        for attr, key in (
            ("prompt_path", "prompt"),
            ("rubric_path", "rubric"),
            ("evidence_path", "evidence"),
            ("meta_path", "meta"),
        ):
            value = getattr(wr, attr, None)
            if value is not None:
                sidecars[key] = str(value)

    rubric_version = rs.rubric_metadata.version if rs.rubric_metadata is not None else 0
    return RefineSummaryDTO(
        keep_count=keep,
        drop_count=drop,
        error_count=err,
        auto_drop_count=auto,
        rubric_version=rubric_version,
        estimated_total_chunks=estimated_total,
        sidecar_paths=sidecars,
        decile_rows=rows,
        total_latency_ms=jr.total_latency_ms,
        operator_decision=rs.operator_decision or "agree",
    )


def reflection_to_dto(reflection: dict[str, Any] | None) -> ReflectionDTO | None:
    if reflection is None:
        return None
    # Strip leading-underscore keys (``_diagnostics`` for raw LLM
    # prompt/response, ``_recommendation_consumed`` for the
    # single-shot recommendation marker).
    safe = {k: v for k, v in reflection.items() if not k.startswith("_")}
    rec = safe.get("path_drop_recommendation")
    rec_dto: PathDropRecommendationDTO | None = None
    # The recommendation is elided once the operator has acted on it,
    # so a refresh after a decision doesn't re-render the banner.
    if rec is not None and not reflection.get("_recommendation_consumed"):
        rec_dto = PathDropRecommendationDTO.model_validate(rec)
    return ReflectionDTO(
        observe=safe.get("observe"),
        diagnose=safe.get("diagnose"),
        hypothesis=safe.get("hypothesis"),
        previous_hypothesis_verdict=safe.get("previous_hypothesis_verdict"),
        path_drop_recommendation=rec_dto,
        status=safe.get("status"),
        turns_to_converge=safe.get("turns_to_converge"),
    )


# ---------------------------------------------------------------------------
# Phase 4 serializers
# ---------------------------------------------------------------------------


def apply_eval_report_to_dto(
    report: Any,
    *,
    eval_scores: list[float] | None = None,
    eval_labels: list[int] | None = None,
) -> ApplyEvalReportDTO:
    return ApplyEvalReportDTO(
        precision_at_threshold=report.precision_at_threshold,
        recall_at_threshold=report.recall_at_threshold,
        pr_curve=[
            PRCurvePointDTO(threshold=t, precision=p, recall=r)
            for (t, p, r) in (report.pr_curve or [])
        ],
        threshold_default=report.threshold_default,
        threshold_selected_by_cv=report.threshold_selected_by_cv,
        cv_precision_mean=report.cv_precision_mean,
        cv_precision_std=report.cv_precision_std,
        min_precision=report.min_precision,
        eval_n=report.eval_n,
        eval_keep_n=report.eval_keep_n,
        eval_drop_n=report.eval_drop_n,
        passes_bar=report.passes_bar,
        eval_scores=list(eval_scores) if eval_scores is not None else [],
        eval_labels=list(eval_labels) if eval_labels is not None else [],
    )


def cohort_projection_to_dto(projection: Any) -> CohortProjectionDTO:
    return CohortProjectionDTO(
        threshold=projection.threshold,
        keep=projection.keep,
        drop=projection.drop,
        total=projection.total,
        per_decile_keep_rate=list(projection.per_decile_keep_rate),
    )


def borderline_samples_to_dtos(samples: list[Any]) -> list[BorderlineSampleDTO]:
    return [
        BorderlineSampleDTO(
            pk=s.pk,
            p_keep=s.p_keep,
            nearest_fit_distance=s.nearest_fit_distance,
            decile=s.decile,
        )
        for s in samples
    ]


def _session_timing(runs_dir: Path, session_id: str) -> tuple[str | None, str | None]:
    """Recover the session's wall-clock start/end from disk sidecars.

    Start comes from ``<sid>.phase2.meta.json`` (``ts`` field written by
    the harvest writer); end comes from ``<sid>.phase4.meta.json``
    (``ts``) when finalize has run, falling back to
    ``<sid>.phase4.eval.json`` (``written_at``) when only Stage A has
    completed. Eval ``written_at`` is slightly earlier than the meta
    ``ts`` because eval is written first — note this when reasoning
    about elapsed duration on a not-yet-finalized session.

    Any read or parse failure leaves the corresponding field ``None``.
    """
    started_at: str | None = None
    ended_at: str | None = None
    try:
        phase2_path = runs_dir / f"{session_id}.phase2.meta.json"
        if phase2_path.exists():
            payload = json.loads(phase2_path.read_text(encoding="utf-8"))
            ts = payload.get("ts")
            if isinstance(ts, str):
                started_at = ts
    except (OSError, ValueError):
        pass

    try:
        phase4_meta_path = runs_dir / f"{session_id}.phase4.meta.json"
        if phase4_meta_path.exists():
            payload = json.loads(phase4_meta_path.read_text(encoding="utf-8"))
            ts = payload.get("ts")
            if isinstance(ts, str):
                ended_at = ts
        if ended_at is None:
            eval_path = runs_dir / f"{session_id}.phase4.eval.json"
            if eval_path.exists():
                payload = json.loads(eval_path.read_text(encoding="utf-8"))
                ts = payload.get("written_at")
                if isinstance(ts, str):
                    ended_at = ts
    except (OSError, ValueError):
        pass

    return started_at, ended_at


def apply_summary_to_dto(
    state: Any,
    *,
    query: str = "",
    runs_dir: Path | None = None,
    session_id: str | None = None,
) -> ApplySummaryDTO:
    """Compose the terminal Phase 4 summary payload.

    Reads the persisted classifier metadata + write result for the
    sidecar paths; falls back to in-memory ``eval_report`` when the
    classifier hasn't been re-loaded from disk.

    When ``runs_dir`` and ``session_id`` are provided, the DTO also
    carries the session's start/end timestamps recovered from disk via
    :func:`_session_timing` (best-effort: missing sidecars leave the
    fields ``None``). ``query`` echoes ``SessionState.query`` so the
    post-apply DONE summary doesn't have to fetch the snapshot
    separately.
    """
    metadata = state.classifier_metadata
    eval_report = state.eval_report
    write = state.write_result

    sidecars: dict[str, str] = {}
    if write is not None:
        for attr, key in (
            ("classifier_path", "classifier"),
            ("eval_path", "eval"),
            ("labels_path", "labels"),
            ("meta_path", "meta"),
        ):
            value = getattr(write, attr, None)
            if value is not None:
                sidecars[key] = str(value)

    final_projection = getattr(state, "final_projection", None)
    if final_projection is not None:
        projection = cohort_projection_to_dto(final_projection)
    else:
        projection = CohortProjectionDTO(
            threshold=metadata.threshold if metadata is not None else 0.0,
            keep=write.n_labels if write is not None and metadata is not None else 0,
            drop=0,
            total=write.n_labels if write is not None else 0,
            per_decile_keep_rate=[],
        )

    started_at: str | None = None
    ended_at: str | None = None
    if runs_dir is not None and session_id is not None:
        started_at, ended_at = _session_timing(runs_dir, session_id)

    return ApplySummaryDTO(
        rubric_version=metadata.rubric_version if metadata is not None else 0,
        threshold=metadata.threshold if metadata is not None else 0.0,
        cohort_projection=projection,
        eval=apply_eval_report_to_dto(eval_report)
        if eval_report is not None
        else ApplyEvalReportDTO(
            precision_at_threshold=0.0,
            recall_at_threshold=0.0,
            pr_curve=[],
            threshold_default=0.0,
            threshold_selected_by_cv=None,
            cv_precision_mean=None,
            cv_precision_std=None,
            min_precision=0.0,
            eval_n=0,
            eval_keep_n=0,
            eval_drop_n=0,
            passes_bar=False,
        ),
        operator_decision=state.operator_decision or "agree",
        sidecar_paths=sidecars,
        class_balance_training={
            "keep": metadata.class_balance.keep if metadata is not None else 0,
            "drop": metadata.class_balance.drop if metadata is not None else 0,
        },
        training_n=len(metadata.training_pks) if metadata is not None else 0,
        eval_metrics_n=len(metadata.eval_pks) if metadata is not None else 0,
        query=query,
        session_started_at=started_at,
        session_ended_at=ended_at,
    )
