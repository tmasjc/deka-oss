// Mirrors src/web_api/schemas.py. Kept hand-written rather than generated so
// the file is a legible contract and shows up in PR diffs.

export type PathName = "dense" | "sparse";
export type Rating = "FIT" | "NOT_FIT" | "DISCARD";
export type WorkflowStatus = "done" | "current" | "pending";

export type Scope = {
  name: string;
  description: string;
  milvus_collection: string;
};

export type EvidenceRow = {
  rank: number;
  pk: string | number;
  chunk_id: string;
  chunk_content: string;
  sample_id: string;
  counselor_id: string;
  term: string;
  source_paths: PathName[];
  scores: Record<PathName, number>;
  rating: Rating | null;
  span_line_indices: number[];
  span_text: string;
};

export type CandidateRow = {
  path: PathName;
  rank_in_path: number;
  pk: string | number;
  chunk_id: string;
  chunk_content: string;
  sample_id: string;
  counselor_id: string;
  term: string;
  score: number;
  rating: Rating | null;
  span_line_indices: number[];
  span_text: string;
};

export type EvidenceTable = {
  query: string;
  rows: EvidenceRow[];
  per_path_candidates: Record<PathName, CandidateRow[]>;
  filtered_short_chunk: number;
  filtered_duplicate_sample: number;
  dropped_by_extractor: number;
};

export type Params = {
  rrf_k: number;
  per_path_limit: number;
  top_k: number;
  active_paths: PathName[];
};

export type Convergence = {
  pk_current: number;
  fit_current: number;
  not_fit_current: number;
  pk_threshold: number;
  fit_threshold: number;
  not_fit_threshold: number;
  converged: boolean;
};

export type WorkflowStep = {
  key: string;
  status: WorkflowStatus;
  label: string;
  detail: string | null;
};

export type BreakdownRow = {
  total: number;
  fit: number;
  not_fit: number;
  discard: number;
};

export type TurnBreakdown = {
  turn: number;
  breakdown: Record<string, BreakdownRow>;
};

export type PathProbeStats = {
  hit_count: number;
  score_min: number | null;
  score_max: number | null;
  score_mean: number | null;
  skipped: boolean;
};

export type ProbeSummary = {
  query: string;
  stats_by_path: Record<PathName, PathProbeStats>;
  rationale: string[];
  flags: string[];
};

export type SessionSnapshot = {
  session_id: string;
  query: string;
  turn_number: number;
  phase: string;
  scope: string;
  table: EvidenceTable;
  params: Params;
  convergence: Convergence;
  workflow: WorkflowStep[];
  breakdown_cumulative: Record<string, BreakdownRow>;
  precision_trend: number[];
  breakdown_by_turn: TurnBreakdown[];
  drop_impact_preview: Record<string, unknown> | null;
  turn_complete: boolean;
  audit_mode_active: boolean;
  probe_summary: ProbeSummary | null;
  // True when the snapshot reflects a DONE_VIEW resume — the UI
  // hides every mutating affordance and the backend 409s every
  // mutating endpoint.
  read_only: boolean;
  // True when the session is being walked through in Replay Mode.
  // Implies ``read_only``; the front-end uses this to swap the
  // standard loaders for the unified "in replay-mode..." overlay and
  // to retarget the "advance" key to the replay-advance endpoint.
  replay: boolean;
};

export type ResumeTarget =
  | "POST_TUNING"
  | "POST_HARVEST"
  | "POST_RUBRIC"
  | "APPLY_PENDING"
  | "DONE_VIEW";

export type SessionListItem = {
  session_id: string;
  query: string;
  scope: string | null;
  resume_target: ResumeTarget;
  last_modified: string;
  n_turns: number;
  has_rubric: boolean;
  has_artifacts: boolean;
};

export type UpdateConfigRequest = {
  rrf_k?: number;
  per_path_limit?: number;
  top_k?: number;
  active_paths?: PathName[];
};

// ---------------------------------------------------------------------------
// Per-session config overrides (Edit parameters modal on QueryEntry)
// ---------------------------------------------------------------------------

// The four phase blocks each accept a curated subset of dynamic knobs;
// fixed/infrastructure values (URLs, models, API keys, paths) are not
// exposed here. The backend's Pydantic validator rejects unknown keys
// with 422, so adding a key here must be paired with a backend update.

export type SearchOverrides = {
  top_k?: number;
  per_path_limit?: number;
  active_paths?: PathName[];
  min_survivors?: number;
};

export type HarvestOverrides = {
  min_fit?: number;
  min_not_fit?: number;
  precision_at_k?: number;
  radius_scheme?: "per_fit" | "decoupled";
  s2c_outlier_multiple?: number;
  anchor_frequency_gate?: number;
};

export type RefineOverrides = {
  sample_size?: number;
  n_bins?: number;
  seed?: number;
  max_fit_examples?: number;
  max_not_fit_examples?: number;
  auto_drop_known_intruders?: boolean;
};

export type ApplyOverrides = {
  enabled?: boolean;
  confidence_threshold?: number;
  min_precision?: number;
  kfold_splits?: number;
};

export type SessionOverrides = {
  search?: SearchOverrides;
  harvest?: HarvestOverrides;
  refine?: RefineOverrides;
  apply?: ApplyOverrides;
};

export type ConfigDefaults = {
  search: Required<SearchOverrides>;
  harvest: Required<HarvestOverrides>;
  refine: Required<RefineOverrides>;
  apply: Required<ApplyOverrides>;
};

export type PathDropRecommendation = {
  path: PathName;
  reason: string;
  confidence: "low" | "medium" | "high";
};

export type Reflection = {
  observe: string | null;
  diagnose: string | null;
  hypothesis: string | null;
  previous_hypothesis_verdict: "CONFIRMED" | "REFUTED" | null;
  path_drop_recommendation: PathDropRecommendation | null;
  status: "CONTINUE" | "CONVERGED" | null;
  turns_to_converge: number | null;
};

export type RecommendationDecision = "apply" | "ignore";

export type DropPathRequest = { path: PathName };

/**
 * A row or per-path candidate, unified for display. The Rating screen
 * cycles through a flat queue: fused rows first (ordered by rank),
 * then per-path candidates in dense → sparse order.
 */
export type RatableItem =
  | { kind: "row"; row: EvidenceRow }
  | { kind: "candidate"; candidate: CandidateRow };

// ---------------------------------------------------------------------------
// Phase 2 (Harvest)
// ---------------------------------------------------------------------------

export type AnchorVerdict = "HEALTHY" | "FLAGGED" | "FAILED";

export type HarvestPreflight = {
  n_fit: number;
  batch_size: number;
  max_k: number;
  radius_scheme: string;
};

export type FrequencyGate = {
  f_configured: number;
  n_fit_after_quality_gate: number;
  kept: number;
  dropped: number;
  qualifying_count_distribution: Record<string, number>;
  qualifying_count_histogram: Record<string, number>;
};

export type QualityGateDrop = {
  fit_chunk_id: string;
  delta: number;
  reasons: string[];
};

export type CohortMissing = { fit_chunk_id: string };

export type HarvestTimings = {
  load_ms: number;
  calibrate_ms: number;
  loo_ms: number;
  retrieve_ms: number;
  total_ms: number;
};

export type AnchorResult = {
  verdict: AnchorVerdict;
  loo_recovered: number;
  loo_total: number;
  T: number;
  delta_min: number;
  delta_median: number;
  delta_max: number;
  T_prime_min: number;
  T_prime_median: number;
  T_prime_max: number;
  T_prime_out: number;
  radius_scheme: string;
  retained_chunks: number;
  not_fit_intrusions: number;
  quality_gate_median_delta_pre_drop: number;
  quality_gate_T_pre_drop: number;
  quality_gate_multiplier: number;
  quality_gate_multiplier_cutoff: number | null;
  quality_gate_median_floor_applied: boolean;
  n_fit_entering_quality_gate: number;
  n_discard_filtered: number;
  frequency_gate: FrequencyGate | null;
  quality_gate_dropped: QualityGateDrop[];
  cohort_consistency_missing: CohortMissing[];
  budget_exhausted: string[];
  sidecar_jsonl_path: string | null;
  sidecar_meta_path: string | null;
  timings: HarvestTimings;
};

// ---------------------------------------------------------------------------
// Phase 3 (Refine — rubric + judge + finalise)
// ---------------------------------------------------------------------------

export type RefinePreflight = {
  phase2_count: number;
  sample_size: number;
  n_bins: number;
  derive_model: string;
  judge_model: string;
};

export type RubricCheck = { id: string; description: string; required: boolean };

export type RubricExample = {
  pk: string | number;
  span_text: string;
  fails: string[] | null;
};

export type RubricMetadata = {
  query: string;
  derive_model_id: string;
  checks: RubricCheck[];
  fit_examples: RubricExample[];
  not_fit_examples: RubricExample[];
  version: number;
};

export type DeriveResult = {
  rubric_text: string;
  metadata: RubricMetadata;
  attempts: number;
  latency_ms: number;
};

export type RubricPrompt = {
  rubric_text: string;
  metadata: RubricMetadata;
};

export type RubricSaveRequest = { rubric_text: string };

export type JudgeDecileBucket = {
  n: number;
  keep: number;
  drop: number;
  error: number;
};

export type JudgeResult = {
  keep_count: number;
  drop_count: number;
  error_count: number;
  parse_error_count: number;
  api_error_count: number;
  total_latency_ms: number;
  decile_breakdown: Record<string, JudgeDecileBucket>;
};

export type VerdictKind = "KEEP" | "DROP" | "ERROR";

export type Verdict = {
  pk: string | number;
  nearest_fit_distance: number;
  decile: number;
  chunk_content: string;
  verdict: VerdictKind;
  evidence_line_indices: number[];
  failed_check: string | null;
  reason: string;
};

export type DecileRow = {
  decile: number;
  distance_min: number | null;
  distance_max: number | null;
  sample_n: number;
  keep_count: number;
  drop_count: number;
  keep_rate: number | null;
};

export type RefineSummary = {
  keep_count: number;
  drop_count: number;
  error_count: number;
  auto_drop_count: number;
  rubric_version: number;
  estimated_total_chunks: number;
  sidecar_paths: Record<string, string>;
  decile_rows: DecileRow[];
  total_latency_ms: number;
  operator_decision: string;
};

// Phase 4 (Apply) DTOs — mirror src/web_api/schemas.py.

export type ApplyPreflight = {
  phase3_finalised: boolean;
  cohort_count: number;
  labels_count: number;
  confidence_threshold: number;
  min_precision: number;
  embedding_dim: number;
};

export type PRCurvePoint = {
  threshold: number;
  precision: number;
  recall: number;
};

export type ApplyEvalReport = {
  precision_at_threshold: number;
  recall_at_threshold: number;
  pr_curve: PRCurvePoint[];
  threshold_default: number;
  threshold_selected_by_cv: number | null;
  cv_precision_mean: number | null;
  cv_precision_std: number | null;
  min_precision: number;
  eval_n: number;
  eval_keep_n: number;
  eval_drop_n: number;
  passes_bar: boolean;
  // Optional on the wire — old backends predate this payload extension
  // and resume-from-disk paths don't persist the raw p_keep vector.
  eval_scores?: number[];
  eval_labels?: number[];
};

export type CohortProjection = {
  threshold: number;
  keep: number;
  drop: number;
  total: number;
  per_decile_keep_rate: (number | null)[];
};

export type BorderlineSample = {
  pk: string | number;
  p_keep: number;
  nearest_fit_distance: number;
  decile: number;
};

export type ApplyCalibrateResponse = {
  projection: CohortProjection;
  borderline_samples: BorderlineSample[];
  eval_at_threshold: ApplyEvalReport;
};

export type ApplyFinalizeRequest = {
  threshold: number;
  allow_low_precision: boolean;
};

export type ApplySummary = {
  rubric_version: number;
  threshold: number;
  cohort_projection: CohortProjection;
  eval: ApplyEvalReport;
  operator_decision: string;
  sidecar_paths: Record<string, string>;
  class_balance_training: Record<string, number>;
  training_n: number;
  eval_metrics_n: number;
  // Display-only echoes consumed by the post-apply DONE summary view.
  // Backend leaves the timing fields null when the source sidecars are
  // missing (legacy sessions, pre-finalize fetches).
  query: string;
  session_started_at: string | null;
  session_ended_at: string | null;
};
