import { http } from "./client";
import type {
  AnchorResult,
  ApplyCalibrateResponse,
  ApplyFinalizeRequest,
  ApplyPreflight,
  ApplySummary,
  ConfigDefaults,
  DeriveResult,
  DropPathRequest,
  HarvestPreflight,
  JudgeResult,
  PathName,
  Rating,
  RecommendationDecision,
  RefinePreflight,
  RefineSummary,
  Reflection,
  RubricMetadata,
  RubricPrompt,
  RubricSaveRequest,
  SessionOverrides,
  SessionSnapshot,
  UpdateConfigRequest,
  Verdict,
} from "../types";

export type NextTurnResponse = {
  snapshot: SessionSnapshot;
  reflection: Reflection | null;
};

export type RatePayload =
  | { rank: number; rating: Rating }
  | { path: PathName; rank_in_path: number; rating: Rating };

export type StartSessionBody = {
  query: string;
  scope: string;
  session_id?: string;
  overrides?: SessionOverrides;
};

export type ProgressDTO = {
  stage: string;
  processed: number;
  total: number | null;
  error: string | null;
  // Free-form per-attempt narration set during multi-attempt operations
  // such as the min_survivors auto-retry. The overlay surfaces this so
  // the operator sees why the wait is longer than usual.
  detail: string | null;
};

export type OriginalContentResponse = {
  pk: number | string;
  original_content: string;
};

export const configApi = {
  defaults: () => http.get<ConfigDefaults>("/api/config/defaults"),
};

export const sessionApi = {
  start: (body: StartSessionBody) =>
    http.post<SessionSnapshot>("/api/session", body),
  get: (sid: string) => http.get<SessionSnapshot>(`/api/session/${sid}`),
  overrides: (sid: string) =>
    http.get<Record<string, Record<string, unknown>>>(
      `/api/session/${sid}/overrides`,
    ),
  rate: (sid: string, payload: RatePayload) =>
    http.post<SessionSnapshot>(`/api/session/${sid}/rate`, payload),
  nextTurn: (sid: string) =>
    http.post<NextTurnResponse>(`/api/session/${sid}/turn/next`),
  reflection: (sid: string) =>
    http.get<Reflection>(`/api/session/${sid}/reflection`),
  editConfig: (sid: string, body: UpdateConfigRequest) =>
    http.post<SessionSnapshot>(`/api/session/${sid}/config`, body),
  triggerAudit: (sid: string) =>
    http.post<SessionSnapshot>(`/api/session/${sid}/audit`),
  dropPath: (sid: string, body: DropPathRequest) =>
    http.post<SessionSnapshot>(`/api/session/${sid}/drop_path`, body),
  recommendationDecision: (
    sid: string,
    body: { decision: RecommendationDecision },
  ) =>
    http.post<SessionSnapshot>(`/api/session/${sid}/recommendation`, body),
  end: (sid: string) => http.del<{ ok: boolean }>(`/api/session/${sid}`),
  progress: (sid: string) =>
    http.get<ProgressDTO>(`/api/session/${sid}/progress`),
  fetchOriginal: (sid: string, pk: number | string) =>
    http.get<OriginalContentResponse>(
      `/api/session/${sid}/chunks/${encodeURIComponent(String(pk))}/original`,
    ),
  harvestPreflight: (sid: string) =>
    http.post<HarvestPreflight>(`/api/session/${sid}/harvest/start`),
  harvestRun: (sid: string) =>
    http.post<SessionSnapshot>(`/api/session/${sid}/harvest/run`, {
      confirm: true,
    }),
  harvestResult: (sid: string) =>
    http.get<AnchorResult>(`/api/session/${sid}/harvest/result`),
  refinePreflight: (sid: string) =>
    http.post<RefinePreflight>(`/api/session/${sid}/refine/start`),
  refineDerive: (sid: string) =>
    http.post<SessionSnapshot>(`/api/session/${sid}/refine/derive`),
  refineDeriveResult: (sid: string) =>
    http.get<DeriveResult>(`/api/session/${sid}/refine/derive_result`),
  refinePrompt: (sid: string) =>
    http.get<RubricPrompt>(`/api/session/${sid}/refine/prompt`),
  refineSaveRubric: (sid: string, body: RubricSaveRequest) =>
    http.post<RubricMetadata>(`/api/session/${sid}/refine/rubric`, body),
  refineJudge: (sid: string) =>
    http.post<SessionSnapshot>(`/api/session/${sid}/refine/judge`),
  refineJudgeResult: (sid: string) =>
    http.get<JudgeResult>(`/api/session/${sid}/refine/judge_result`),
  refineVerdicts: (sid: string) =>
    http.get<Verdict[]>(`/api/session/${sid}/refine/verdicts`),
  refineDiscard: (sid: string) =>
    http.post<RubricMetadata>(`/api/session/${sid}/refine/discard`),
  refineFinalize: (sid: string) =>
    http.post<RefineSummary>(`/api/session/${sid}/refine/finalize`),
  refineSummaryResult: (sid: string) =>
    http.get<RefineSummary>(`/api/session/${sid}/refine/summary`),
  applyPreflight: (sid: string) =>
    http.post<ApplyPreflight>(`/api/session/${sid}/apply/start`),
  applyTrain: (sid: string) =>
    http.post<SessionSnapshot>(`/api/session/${sid}/apply/train`),
  applyEval: (sid: string, threshold?: number) => {
    const path =
      threshold === undefined
        ? `/api/session/${sid}/apply/eval`
        : `/api/session/${sid}/apply/eval?threshold=${threshold}`;
    return http.get<ApplyCalibrateResponse>(path);
  },
  applyFinalize: (sid: string, body: ApplyFinalizeRequest) =>
    http.post<ApplySummary>(`/api/session/${sid}/apply/finalize`, body),
  applyCancel: (sid: string) =>
    http.post<SessionSnapshot>(`/api/session/${sid}/apply/cancel`),
  applySummary: (sid: string) =>
    http.get<ApplySummary>(`/api/session/${sid}/apply/summary`),
};

// Pre-flight (issue #33) — env / config sanity checks before session start.
export type PreflightCheck = {
  name: string;
  status: "ok" | "fail";
  detail: string;
  code: string;
  env_var: string;
};

export type PreflightResponse = {
  checks: PreflightCheck[];
  all_passed: boolean;
};

// 4xx body shape for failures — mirrors HTTPException.detail in
// src/web_api/app.py's preflight handler.
export type PreflightFailureDetail = {
  code: string;
  phase: string;
  env_var: string;
  detail: string;
  checks: PreflightCheck[];
};

export const preflightApi = {
  check: (scope: string) =>
    http.post<PreflightResponse>("/api/session/preflight", { scope }),
};

export type AuthMeResponse = { user_id: string };

export const authApi = {
  login: (token: string) =>
    http.post<AuthMeResponse>("/api/auth/login", { token }),
  logout: () => http.post<void>("/api/auth/logout"),
  me: () => http.get<AuthMeResponse>("/api/auth/me"),
};

import type { SessionListItem } from "../types";

export const sessionsApi = {
  list: () => http.get<SessionListItem[]>("/api/sessions"),
  resume: (sid: string) =>
    http.post<SessionSnapshot>(`/api/session/${sid}/resume`),
  discard: (sid: string) =>
    http.post<{ ok: boolean }>(`/api/session/${sid}/discard`),
  replayStart: (sid: string) =>
    http.post<SessionSnapshot>(`/api/session/${sid}/replay`),
  replayAdvance: (sid: string) =>
    http.post<SessionSnapshot>(`/api/session/${sid}/replay/advance`),
};
