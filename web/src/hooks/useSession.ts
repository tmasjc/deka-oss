import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  sessionApi,
  type RatePayload,
  type StartSessionBody,
} from "../api/session";
import type {
  DropPathRequest,
  RecommendationDecision,
  SessionSnapshot,
  UpdateConfigRequest,
} from "../types";
import { useUi } from "../state/ui";

const sessionKey = (sid: string) => ["session", sid];

export function useSession(sid: string | null) {
  return useQuery({
    queryKey: sid ? sessionKey(sid) : ["session", "none"],
    queryFn: () => sessionApi.get(sid!),
    enabled: !!sid,
    staleTime: Infinity,
    // Phases where the backend flips state from a worker thread with no
    // client-driven trigger (apply training/applying). Without polling
    // the snapshot stays APPLY_TRAINING and the UI shows the spinner
    // until a manual page reload.
    refetchInterval: (q) => {
      const phase = (q.state.data as SessionSnapshot | undefined)?.phase;
      return phase === "APPLY_TRAINING" ||
        phase === "APPLY_PREPARING" ||
        phase === "APPLY_APPLYING"
        ? 500
        : false;
    },
  });
}

export function useStartSession() {
  const qc = useQueryClient();
  const setSession = useUi((s) => s.setSession);
  return useMutation({
    mutationFn: (body: StartSessionBody) => sessionApi.start(body),
    onSuccess: (snapshot: SessionSnapshot) => {
      qc.setQueryData(sessionKey(snapshot.session_id), snapshot);
      // Invalidate the sessions listing so a return-to-list shows
      // the new session.
      qc.invalidateQueries({ queryKey: ["sessions", "list"] });
      setSession(snapshot.session_id);
      window.history.pushState(null, "", `/s/${snapshot.session_id}`);
    },
  });
}

export function useRate(sid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (payload: RatePayload) => sessionApi.rate(sid, payload),
    onSuccess: (snapshot: SessionSnapshot) => {
      qc.setQueryData(sessionKey(sid), snapshot);
    },
  });
}

export function useNextTurn(sid: string) {
  const qc = useQueryClient();
  const resetCursor = useUi((s) => s.resetCursor);
  return useMutation({
    mutationFn: () => sessionApi.nextTurn(sid),
    onSuccess: (res) => {
      qc.setQueryData(sessionKey(sid), res.snapshot);
      resetCursor();
    },
  });
}

export function useEditConfig(sid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: UpdateConfigRequest) =>
      sessionApi.editConfig(sid, body),
    onSuccess: (snapshot: SessionSnapshot) => {
      qc.setQueryData(sessionKey(sid), snapshot);
    },
  });
}

export function useTriggerAudit(sid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => sessionApi.triggerAudit(sid),
    onSuccess: (snapshot: SessionSnapshot) => {
      qc.setQueryData(sessionKey(sid), snapshot);
    },
  });
}

export function useDropPath(sid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: DropPathRequest) => sessionApi.dropPath(sid, body),
    onSuccess: (snapshot: SessionSnapshot) => {
      qc.setQueryData(sessionKey(sid), snapshot);
    },
  });
}

export function useRecommendationDecision(sid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { decision: RecommendationDecision }) =>
      sessionApi.recommendationDecision(sid, body),
    onSuccess: (snapshot: SessionSnapshot) => {
      qc.setQueryData(sessionKey(sid), snapshot);
    },
  });
}

export function useEndSession() {
  const qc = useQueryClient();
  const setSession = useUi((s) => s.setSession);
  return useMutation({
    mutationFn: (sid: string) => sessionApi.end(sid),
    onSuccess: (_res, sid) => {
      qc.removeQueries({ queryKey: sessionKey(sid) });
      // Refresh the listing so the now-evicted session shows the
      // most recent activity timestamp on next render.
      qc.invalidateQueries({ queryKey: ["sessions", "list"] });
      setSession(null);
      window.history.pushState(null, "", "/");
    },
  });
}
