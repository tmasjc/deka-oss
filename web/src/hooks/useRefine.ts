import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { sessionApi } from "../api/session";
import type {
  DeriveResult,
  JudgeResult,
  RefinePreflight,
  RefineSummary,
  RubricMetadata,
  RubricPrompt,
  RubricSaveRequest,
  SessionSnapshot,
  Verdict,
} from "../types";

const sessionKey = (sid: string) => ["session", sid];
const deriveResultKey = (sid: string) => ["refine-derive-result", sid];
const judgeResultKey = (sid: string) => ["refine-judge-result", sid];
const verdictsKey = (sid: string) => ["refine-verdicts", sid];
const summaryKey = (sid: string) => ["refine-summary", sid];

export function useRefinePreflight(sid: string) {
  return useMutation({
    mutationFn: (): Promise<RefinePreflight> =>
      sessionApi.refinePreflight(sid),
  });
}

export function useRefineDerive(sid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => sessionApi.refineDerive(sid),
    onSuccess: (snapshot: SessionSnapshot) => {
      qc.setQueryData(sessionKey(sid), snapshot);
    },
  });
}

export function useDeriveResult(sid: string, enabled: boolean) {
  return useQuery<DeriveResult>({
    queryKey: deriveResultKey(sid),
    queryFn: () => sessionApi.refineDeriveResult(sid),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}

export function useRubricPrompt(sid: string, enabled: boolean) {
  return useQuery<RubricPrompt>({
    queryKey: ["refine-prompt", sid],
    queryFn: () => sessionApi.refinePrompt(sid),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}

export function useSaveRubric(sid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: RubricSaveRequest) => sessionApi.refineSaveRubric(sid, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: deriveResultKey(sid) });
      qc.invalidateQueries({ queryKey: judgeResultKey(sid) });
      qc.invalidateQueries({ queryKey: verdictsKey(sid) });
    },
  });
}

export function useRefineJudge(sid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => sessionApi.refineJudge(sid),
    onSuccess: (snapshot: SessionSnapshot) => {
      qc.setQueryData(sessionKey(sid), snapshot);
    },
  });
}

export function useJudgeResult(sid: string, enabled: boolean) {
  return useQuery<JudgeResult>({
    queryKey: judgeResultKey(sid),
    queryFn: () => sessionApi.refineJudgeResult(sid),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}

// Pull the Phase 3 summary from disk for an already-finalised session.
// The live finalize flow already returns the same DTO from the POST,
// so this hook only fires for resumes (DONE_VIEW) where the React
// component never saw the original mutation response.
export function useRefineSummary(sid: string, enabled: boolean) {
  return useQuery<RefineSummary>({
    queryKey: summaryKey(sid),
    queryFn: () => sessionApi.refineSummaryResult(sid),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}

export function useVerdicts(sid: string, enabled: boolean) {
  return useQuery<Verdict[]>({
    queryKey: verdictsKey(sid),
    queryFn: () => sessionApi.refineVerdicts(sid),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}

export function useRefineDiscard(sid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (): Promise<RubricMetadata> => sessionApi.refineDiscard(sid),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: sessionKey(sid) });
      qc.invalidateQueries({ queryKey: verdictsKey(sid) });
      qc.invalidateQueries({ queryKey: judgeResultKey(sid) });
    },
  });
}

export function useFinalize(sid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (): Promise<RefineSummary> => sessionApi.refineFinalize(sid),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: sessionKey(sid) });
    },
  });
}

export const refineKeys = {
  derive: deriveResultKey,
  judge: judgeResultKey,
  verdicts: verdictsKey,
};
