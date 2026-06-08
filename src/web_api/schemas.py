"""Pydantic request/response DTOs for the web API.

These mirror the dataclass shapes in :mod:`src.search.evidence` and
:mod:`src.session.state` but are explicit, typed, and JSON-first so the
frontend has a stable contract. Conversion lives in :mod:`serialize`.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

PathName = Literal["dense", "sparse"]
Rating = Literal["FIT", "NOT_FIT", "DISCARD"]
# Workflow keys are freeform to allow per-turn ids like "TURN_1", "TURN_2"
# alongside the fixed "START" / "CONVERGED" / "HARVEST" nodes.
WorkflowStatus = Literal["done", "current", "pending"]


class EvidenceRowDTO(BaseModel):
    rank: int
    pk: Union[int, str]
    chunk_id: str
    chunk_content: str
    sample_id: str
    counselor_id: str
    term: str
    source_paths: list[PathName]
    scores: dict[PathName, float]
    rating: Rating | None = None
    span_line_indices: list[int] = Field(default_factory=list)
    span_text: str = ""


class CandidateRowDTO(BaseModel):
    path: PathName
    rank_in_path: int
    pk: Union[int, str]
    chunk_id: str
    chunk_content: str
    sample_id: str
    counselor_id: str
    term: str
    score: float
    rating: Rating | None = None
    span_line_indices: list[int] = Field(default_factory=list)
    span_text: str = ""


class EvidenceTableDTO(BaseModel):
    query: str
    rows: list[EvidenceRowDTO]
    per_path_candidates: dict[PathName, list[CandidateRowDTO]]
    filtered_short_chunk: int = 0
    filtered_duplicate_sample: int = 0
    dropped_by_extractor: int = 0


class ParamsDTO(BaseModel):
    rrf_k: int
    per_path_limit: int
    top_k: int
    active_paths: list[PathName]


class ConvergenceDTO(BaseModel):
    pk_current: float
    fit_current: int
    not_fit_current: int
    pk_threshold: float
    fit_threshold: int
    not_fit_threshold: int
    converged: bool


class WorkflowStepDTO(BaseModel):
    key: str
    status: WorkflowStatus
    label: str
    detail: str | None = None


class PathProbeStatsDTO(BaseModel):
    """Per-path Turn-0 probe stats (mirrors ``ProbeResult.stats_by_path[path]``)."""

    hit_count: int = 0
    score_min: float | None = None
    score_max: float | None = None
    score_mean: float | None = None
    skipped: bool = False


class ProbeSummaryDTO(BaseModel):
    """Turn-0 probe + adapt diagnostics surfaced as a banner.

    Present on the snapshot only on Turn 1.
    """

    query: str
    stats_by_path: dict[PathName, PathProbeStatsDTO]
    rationale: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class BreakdownRowDTO(BaseModel):
    total: int
    fit: int
    not_fit: int
    discard: int = 0


class TurnBreakdownDTO(BaseModel):
    turn: int
    breakdown: dict[str, BreakdownRowDTO]


class SessionSnapshot(BaseModel):
    """Everything the Rating screen needs in a single payload."""

    session_id: str
    query: str
    turn_number: int
    phase: str
    scope: str
    table: EvidenceTableDTO
    params: ParamsDTO
    convergence: ConvergenceDTO
    workflow: list[WorkflowStepDTO]
    breakdown_cumulative: dict[str, BreakdownRowDTO]
    precision_trend: list[float] = Field(default_factory=list)
    breakdown_by_turn: list[TurnBreakdownDTO] = Field(default_factory=list)
    drop_impact_preview: dict[str, Any] | None = None
    # True when the user has rated every row/candidate in the current turn.
    # Front-end uses this to decide whether pressing `a` is valid.
    turn_complete: bool
    read_only: bool = False
    # True when the session is being walked through in Replay Mode —
    # a time-traveller view that surfaces historical turn/phase state
    # without mutating any sidecar. ``read_only`` is also true when
    # this is set; the flag exists so the frontend can swap loaders /
    # disable affordances explicitly.
    replay: bool = False
    # True when the operator has triggered an audit on the current
    # (in-progress) turn — per-path candidates are then expected for
    # rating and ``POST /drop_path`` becomes valid.
    audit_mode_active: bool = False
    # Turn-0 probe diagnostics; the front-end renders these as a
    # dismissible banner on Turn 1. Cleared (None) on subsequent turns.
    probe_summary: ProbeSummaryDTO | None = None


# Per-session override allow-list. Every key listed here is a knob that
# affects results or cost; nothing in this set is infrastructure (URLs,
# DSNs, API keys, model ids, paths, embed dim, prompt versions). The
# allow-list is the single security boundary that stops a client from
# overriding fixed parameters by piggy-backing on a YAML section name —
# anything outside the list returns 422 before reaching the YAML loader.
_OVERRIDE_ALLOWLIST: dict[str, frozenset[str]] = {
    "search": frozenset({"top_k", "per_path_limit", "active_paths", "min_survivors"}),
    "harvest": frozenset(
        {
            "min_fit",
            "min_not_fit",
            "precision_at_k",
            "radius_scheme",
            "s2c_outlier_multiple",
            "anchor_frequency_gate",
        }
    ),
    "refine": frozenset(
        {
            "sample_size",
            "n_bins",
            "seed",
            "max_fit_examples",
            "max_not_fit_examples",
            "auto_drop_known_intruders",
        }
    ),
    "apply": frozenset(
        {"enabled", "confidence_threshold", "min_precision", "kfold_splits"}
    ),
}


def override_allowlist() -> dict[str, frozenset[str]]:
    """Public accessor — used by `/api/config/defaults` to project YAML."""
    return _OVERRIDE_ALLOWLIST


class SessionOverrides(BaseModel):
    """Per-session overrides, scoped by phase section.

    Each phase block is an optional dict whose keys must be in the
    curated allow-list. The Pydantic validator below enforces that any
    unknown key (e.g. ``milvus_uri``, ``judge_model``) fails the request
    with 422 — fixed/infrastructure values stay in ``config.yaml`` only.
    """

    search: dict[str, Any] | None = None
    harvest: dict[str, Any] | None = None
    refine: dict[str, Any] | None = None
    apply: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _enforce_allowlist(self) -> "SessionOverrides":
        for section, allowed in _OVERRIDE_ALLOWLIST.items():
            block = getattr(self, section)
            if block is None:
                continue
            if not isinstance(block, dict):
                raise ValueError(
                    f"overrides.{section} must be an object; got {type(block).__name__}"
                )
            unknown = set(block.keys()) - allowed
            if unknown:
                raise ValueError(
                    f"overrides.{section} contains disallowed keys: {sorted(unknown)}. "
                    f"Allowed: {sorted(allowed)}"
                )
        return self

    def to_sidecar_dict(self) -> dict[str, dict[str, Any]]:
        """Return the non-empty sections as plain dicts for JSON dump."""
        out: dict[str, dict[str, Any]] = {}
        for section in _OVERRIDE_ALLOWLIST:
            block = getattr(self, section)
            if block:
                out[section] = dict(block)
        return out


class StartSessionRequest(BaseModel):
    query: str
    scope: str
    # Optional client-generated id. When provided, the server uses it as
    # the session id so the client can poll /progress before the POST returns.
    session_id: str | None = None
    # Optional per-session config overrides set via the query page's
    # [Edit parameters] modal. None / empty == use config.yaml defaults.
    overrides: SessionOverrides | None = None


class ScopeDTO(BaseModel):
    name: str
    description: str
    milvus_collection: str


class ScopesResponse(BaseModel):
    scopes: list[ScopeDTO]


class AuthLoginRequest(BaseModel):
    """Body for ``POST /api/auth/login``.

    The ``token`` is the plaintext bearer the user received out of band.
    The server hashes it (SHA-256) and looks the digest up in
    ``users.yaml`` — the plaintext is never persisted server-side.
    """

    token: str


class AuthMeResponse(BaseModel):
    """Body for ``POST /api/auth/login`` and ``GET /api/auth/me``."""

    user_id: str


class SessionListItem(BaseModel):
    """One row in the post-login session picker.

    ``resume_target`` is the badge the row renders. ``query`` and
    ``scope`` are static-per-session metadata read from the
    canonical jsonl's first turn row. Sessions that classify as
    abandoned never appear in the list.
    """

    session_id: str
    query: str
    scope: str | None
    resume_target: str  # ResumeTarget value
    last_modified: str  # ISO 8601, UTC, second resolution
    n_turns: int
    has_rubric: bool
    has_artifacts: bool


class ProgressDTO(BaseModel):
    """Live progress snapshot for the in-flight bootstrap/advance operation."""

    stage: str
    processed: int = 0
    total: int | None = None
    error: str | None = None
    # Free-form per-attempt narration (e.g. "Re-fusing with per_path_limit
    # 40 — only 2 survivors after filter"). Set during multi-attempt
    # operations so the UI can surface why the wait is longer than usual.
    detail: str | None = None


class UpdateConfigRequest(BaseModel):
    """Partial config override. At least one field must be present."""

    rrf_k: int | None = None
    per_path_limit: int | None = None
    top_k: int | None = None
    active_paths: list[PathName] | None = None


class RateRequest(BaseModel):
    """Either (rank) for a fused row or (path + rank_in_path) for a candidate."""

    rank: int | None = None
    path: PathName | None = None
    rank_in_path: int | None = None
    rating: Rating


class PathDropRecommendationDTO(BaseModel):
    """Optional structured path-drop nomination from reflection.

    Mirrors :class:`src.reflection.schema.PathDropRecommendation`. On
    ``apply`` the path is dropped immediately via the recommendation
    endpoint — no audit step, no Rule B at the apply site.
    """

    path: PathName
    reason: str
    confidence: Literal["low", "medium", "high"]


class ReflectionDTO(BaseModel):
    """Reasoning trace for a completed turn.

    Reflection no longer prescribes a config — the session config is
    locked once turn 1 starts. The agent's only lever is the optional
    ``path_drop_recommendation``; on operator apply the drop is applied
    directly.
    """

    observe: str | None = None
    diagnose: str | None = None
    hypothesis: str | None = None
    previous_hypothesis_verdict: Literal["CONFIRMED", "REFUTED"] | None = None
    path_drop_recommendation: PathDropRecommendationDTO | None = None
    status: Literal["CONTINUE", "CONVERGED"] | None = None
    turns_to_converge: int | None = None


class RecommendationDecisionRequest(BaseModel):
    """Operator's response to a path-drop recommendation.

    ``apply``  — drop the recommended path from ``active_paths``
                 immediately (no audit, no Rule B at apply site).
    ``ignore`` — log the decision; no state change.
    """

    decision: Literal["apply", "ignore"]


class DropPathRequest(BaseModel):
    """Path the operator wants to drop after an audit.

    The web API checks audit_mode_active + all_rated, then delegates
    to ``SessionState.apply_path_drop`` (Rule B). Rule B failures
    surface as 409 with the rejection reason; the session config is
    unchanged on rejection.
    """

    path: PathName


class NextTurnResponse(BaseModel):
    snapshot: SessionSnapshot
    reflection: ReflectionDTO | None = None


class OkResponse(BaseModel):
    ok: bool = True


# ---------------------------------------------------------------------------
# Phase 2 (Harvest)
# ---------------------------------------------------------------------------


class HarvestPreflightDTO(BaseModel):
    """Snapshot of what a harvest run would consume — shown in the
    confirm modal before kicking off the worker.
    """

    n_fit: int
    batch_size: int
    max_k: int
    radius_scheme: str


class HarvestRunRequest(BaseModel):
    """Operator's confirmation to start the harvest worker.

    The body is required (rather than a bare ``POST``) so the request
    contract stays explicit and forward-compatible with future
    overrides like ``allow_unconverged`` or ``dry_run``.
    """

    confirm: bool = True


class FrequencyGateDTO(BaseModel):
    f_configured: int
    n_fit_after_quality_gate: int
    kept: int
    dropped: int
    qualifying_count_distribution: dict[str, int]
    qualifying_count_histogram: dict[str, int] = Field(default_factory=dict)


class QualityGateDropDTO(BaseModel):
    fit_chunk_id: str
    delta: float
    reasons: list[str]


class CohortMissingDTO(BaseModel):
    fit_chunk_id: str


class HarvestTimingsDTO(BaseModel):
    load_ms: float
    calibrate_ms: float
    loo_ms: float
    retrieve_ms: float
    total_ms: float


class AnchorResultDTO(BaseModel):
    """Flat, JSON-friendly view of :class:`src.anchor.runner.AnchorResult`."""

    verdict: Literal["HEALTHY", "FLAGGED", "FAILED"]
    loo_recovered: int
    loo_total: int
    T: float
    delta_min: float
    delta_median: float
    delta_max: float
    T_prime_min: float
    T_prime_median: float
    T_prime_max: float
    T_prime_out: float
    radius_scheme: str
    retained_chunks: int
    not_fit_intrusions: int
    # Quality-gate diagnostics (issues #47, #48). Under the floored-
    # median logic (#47) ``multiplier_cutoff`` is always populated;
    # ``quality_gate_median_floor_applied`` distinguishes the two
    # regimes (median ≥ floor → cutoff tracks median; median < floor
    # → cutoff backstopped by ``_MEDIAN_DELTA_FLOOR``). A ``None``
    # cutoff signals a legacy sidecar from before #47 where the rule
    # was disabled outright.
    quality_gate_median_delta_pre_drop: float = 0.0
    quality_gate_T_pre_drop: float = 0.0
    quality_gate_multiplier: float = 0.0
    quality_gate_multiplier_cutoff: float | None = None
    quality_gate_median_floor_applied: bool = False
    n_fit_entering_quality_gate: int = 0
    n_discard_filtered: int = 0
    frequency_gate: FrequencyGateDTO | None = None
    quality_gate_dropped: list[QualityGateDropDTO] = Field(default_factory=list)
    cohort_consistency_missing: list[CohortMissingDTO] = Field(default_factory=list)
    budget_exhausted: list[str] = Field(default_factory=list)
    sidecar_jsonl_path: str | None = None
    sidecar_meta_path: str | None = None
    timings: HarvestTimingsDTO


# ---------------------------------------------------------------------------
# Phase 3 (Refine — rubric, sampling, judging, finalisation)
# ---------------------------------------------------------------------------


class RefinePreflightDTO(BaseModel):
    """Snapshot of what a refine run will consume — shown in the
    confirm modal before kicking off derive.
    """

    phase2_count: int
    sample_size: int
    n_bins: int
    derive_model: str
    judge_model: str


class RubricCheckDTO(BaseModel):
    id: str
    description: str
    required: bool = True


class RubricExampleDTO(BaseModel):
    """Either a FIT example (``fails`` is None) or a NOT_FIT example
    (``fails`` lists the check ids the example violates).
    """

    pk: Union[int, str]
    span_text: str
    fails: list[str] | None = None


class RubricMetadataDTO(BaseModel):
    query: str
    derive_model_id: str
    checks: list[RubricCheckDTO]
    fit_examples: list[RubricExampleDTO]
    not_fit_examples: list[RubricExampleDTO]
    version: int


class DeriveResultDTO(BaseModel):
    rubric_text: str
    metadata: RubricMetadataDTO
    attempts: int
    latency_ms: float


class RubricPromptDTO(BaseModel):
    """The shipped rubric text + parsed metadata, decoupled from the
    derive run. Survives DONE_VIEW resume (re-hydrated from the
    ``.phase3.prompt.md`` / ``.phase3.rubric.json`` sidecars), unlike
    :class:`DeriveResultDTO` which needs an in-memory ``derive_result``.
    """

    rubric_text: str
    metadata: RubricMetadataDTO


class RubricSaveRequest(BaseModel):
    rubric_text: str


class JudgeDecileBucketDTO(BaseModel):
    n: int
    keep: int
    drop: int
    error: int


class JudgeResultDTO(BaseModel):
    keep_count: int
    drop_count: int
    error_count: int
    parse_error_count: int
    api_error_count: int
    total_latency_ms: float
    decile_breakdown: dict[str, JudgeDecileBucketDTO]


class VerdictDTO(BaseModel):
    pk: Union[int, str]
    nearest_fit_distance: float
    decile: int
    chunk_content: str
    verdict: Literal["KEEP", "DROP", "ERROR"]
    evidence_line_indices: list[int] = Field(default_factory=list)
    failed_check: str | None = None
    reason: str


class DecileRowDTO(BaseModel):
    """One row of the Phase 3 sample-distribution table.

    ``sample_n`` is the cohort population in the bin (from
    ``StratifiedSample.per_decile_count``); ``keep_count`` /
    ``drop_count`` are the population projections built from the
    sample's keep-rate × population count, matching the TUI's
    :class:`RefineSummaryScreen` table.
    """

    decile: int  # 1-based for display
    distance_min: float | None = None
    distance_max: float | None = None
    sample_n: int
    keep_count: int
    drop_count: int
    keep_rate: float | None = None


class RefineSummaryDTO(BaseModel):
    keep_count: int
    drop_count: int
    error_count: int
    auto_drop_count: int
    rubric_version: int
    estimated_total_chunks: int
    sidecar_paths: dict[str, str]
    decile_rows: list[DecileRowDTO]
    total_latency_ms: float
    operator_decision: str


class OriginalContentResponse(BaseModel):
    """Original (full) chunk text fetched from the scope-routed Postgres table."""

    pk: Union[int, str]
    original_content: str


# ---------------------------------------------------------------------------
# Phase 4 (Apply) DTOs
# ---------------------------------------------------------------------------


class ApplyPreflightDTO(BaseModel):
    """Snapshot of what a Phase 4 run will consume — shown in the
    confirm modal before kicking off training.
    """

    phase3_finalised: bool
    cohort_count: int
    labels_count: int
    confidence_threshold: float
    min_precision: float
    embedding_dim: int


class BorderlineSampleDTO(BaseModel):
    pk: Union[int, str]
    p_keep: float
    nearest_fit_distance: float
    decile: int


class PRCurvePointDTO(BaseModel):
    threshold: float
    precision: float
    recall: float


class ApplyEvalReportDTO(BaseModel):
    """The eval-split metrics + PR curve produced by training.

    Returned from ``GET /apply/eval`` and embedded in the calibrate
    response so the web UI's threshold slider has everything it needs.
    """

    precision_at_threshold: float
    recall_at_threshold: float
    pr_curve: list[PRCurvePointDTO]
    threshold_default: float
    threshold_selected_by_cv: float | None
    cv_precision_mean: float | None
    cv_precision_std: float | None
    min_precision: float
    eval_n: int
    eval_keep_n: int
    eval_drop_n: int
    passes_bar: bool
    # Raw eval-set scores + labels for the score-distribution histogram.
    # ~100 floats / ints per response — negligible payload, lets the
    # client redraw the histogram instantly when τ moves without an
    # extra round-trip. Empty when no eval split exists (resume from
    # disk; we don't persist the raw p_keep vector).
    eval_scores: list[float] = []
    eval_labels: list[int] = []


class CohortProjectionDTO(BaseModel):
    threshold: float
    keep: int
    drop: int
    total: int
    per_decile_keep_rate: list[float | None]


class ApplyCalibrateResponse(BaseModel):
    """Live projection at a candidate threshold + borderline samples."""

    projection: CohortProjectionDTO
    borderline_samples: list[BorderlineSampleDTO]
    eval_at_threshold: ApplyEvalReportDTO


class ApplyFinalizeRequest(BaseModel):
    """Body for POST /apply/finalize."""

    threshold: float
    allow_low_precision: bool = False


class ApplySummaryDTO(BaseModel):
    """Final Phase 4 summary surfaced after finalize and on DONE view."""

    rubric_version: int
    threshold: float
    cohort_projection: CohortProjectionDTO
    eval: ApplyEvalReportDTO
    operator_decision: str
    sidecar_paths: dict[str, str]
    class_balance_training: dict[str, int]
    training_n: int
    eval_metrics_n: int
    # Display-only fields consumed by the post-apply DONE summary screen.
    # ``query`` echoes the session's original query so the UI doesn't
    # have to cross-reference the snapshot. Timing fields bracket the
    # session's wall-clock duration and are ``None`` when the source
    # sidecar is absent (e.g. legacy sessions, or summary fetched
    # before finalize wrote phase4.meta.json).
    query: str = ""
    session_started_at: str | None = None
    session_ended_at: str | None = None


# ---------------------------------------------------------------------------
# Pre-flight (issue #33) — env / config sanity checks before session start
# ---------------------------------------------------------------------------


class PreflightCheckDTO(BaseModel):
    """One row in the pre-flight result list.

    ``code`` and ``env_var`` are empty strings on success; on failure
    ``code`` is the machine-readable label the UI keys off (e.g.
    ``MISSING_LLM_KEY``) and ``env_var`` names the offending env var so
    the operator sees exactly what to set.
    """

    name: str
    status: Literal["ok", "fail"]
    detail: str = ""
    code: str = ""
    env_var: str = ""


class PreflightRequest(BaseModel):
    """Body for ``POST /api/session/preflight``.

    The scope drives the Milvus collection check; query is not
    needed at pre-flight time (no probe is run) but the UI passes it
    forward to ``POST /api/session`` once checks pass.
    """

    scope: str


class PreflightResponse(BaseModel):
    """Body returned on a green pre-flight (200).

    Failures surface as a 400 with ``{code, phase, env_var, detail,
    checks}`` in the response body — different shape because FastAPI's
    HTTPException carries the failing-check fields at the top level for
    convenient UI access.
    """

    checks: list[PreflightCheckDTO]
    all_passed: bool = True
