import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { sessionApi } from "../api/session";
import type {
  ApplyCalibrateResponse,
  ApplyFinalizeRequest,
  ApplyPreflight,
  ApplySummary,
  SessionSnapshot,
} from "../types";

const sessionKey = (sid: string) => ["session", sid];
const applyEvalKey = (sid: string, threshold?: number) =>
  threshold === undefined
    ? ["apply-eval", sid]
    : ["apply-eval", sid, threshold];
const applySummaryKey = (sid: string) => ["apply-summary", sid];

export function useApplyPreflight(sid: string) {
  return useMutation({
    mutationFn: (): Promise<ApplyPreflight> => sessionApi.applyPreflight(sid),
  });
}

export function useApplyTrain(sid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => sessionApi.applyTrain(sid),
    onSuccess: (snap: SessionSnapshot) => {
      qc.setQueryData(sessionKey(sid), snap);
    },
  });
}

// Polled while training is in flight (returns 409) and again after the
// session reaches APPLY_REVIEW. Each slider movement fires a new query
// keyed on the threshold so React Query caches per-value.
export function useApplyEval(
  sid: string,
  enabled: boolean,
  threshold?: number,
) {
  return useQuery<ApplyCalibrateResponse>({
    queryKey: applyEvalKey(sid, threshold),
    queryFn: () => sessionApi.applyEval(sid, threshold),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}

export function useApplyFinalize(sid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ApplyFinalizeRequest): Promise<ApplySummary> =>
      sessionApi.applyFinalize(sid, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: sessionKey(sid) });
      qc.invalidateQueries({ queryKey: applySummaryKey(sid) });
    },
  });
}

export function useApplyCancel(sid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => sessionApi.applyCancel(sid),
    onSuccess: (snap: SessionSnapshot) => {
      qc.setQueryData(sessionKey(sid), snap);
    },
  });
}

export function useApplySummary(sid: string, enabled: boolean) {
  return useQuery<ApplySummary>({
    queryKey: applySummaryKey(sid),
    queryFn: () => sessionApi.applySummary(sid),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}

export const applyKeys = {
  evaluate: applyEvalKey,
  summary: applySummaryKey,
};
