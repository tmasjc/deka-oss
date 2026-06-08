import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { sessionsApi } from "../api/session";
import type { SessionListItem, SessionSnapshot } from "../types";
import { useUi } from "../state/ui";

const sessionsKey = ["sessions", "list"] as const;
const sessionKey = (sid: string) => ["session", sid];

/** Listing of the calling user's resumable sessions. The session
 * picker calls this on mount; ``staleTime: 0`` so a fresh sign-out
 * + sign-in yields a fresh fetch. */
export function useSessions() {
  return useQuery<SessionListItem[]>({
    queryKey: sessionsKey,
    queryFn: () => sessionsApi.list(),
    staleTime: 0,
  });
}

/** Hydrate a previously-quit session and route to ``/s/<sid>``. */
export function useResume() {
  const qc = useQueryClient();
  const setSession = useUi((s) => s.setSession);
  return useMutation<SessionSnapshot, Error, string>({
    mutationFn: (sid: string) => sessionsApi.resume(sid),
    onSuccess: (snapshot: SessionSnapshot) => {
      qc.setQueryData(sessionKey(snapshot.session_id), snapshot);
      setSession(snapshot.session_id);
      window.history.pushState(null, "", `/s/${snapshot.session_id}`);
    },
  });
}

/** Hard-delete a session: server removes all on-disk sidecars and the
 * row vanishes from the listing on success via cache invalidation. */
export function useDiscardSession() {
  const qc = useQueryClient();
  return useMutation<{ ok: boolean }, Error, string>({
    mutationFn: (sid: string) => sessionsApi.discard(sid),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: sessionsKey });
    },
  });
}

/** Open a past session in Replay Mode and route to ``/s/<sid>``.
 *
 * Lands at Phase 1, turn 1 read-only; the user then advances with
 * ``a`` (or the next button), which calls :func:`useReplayAdvance`. */
export function useStartReplay() {
  const qc = useQueryClient();
  const setSession = useUi((s) => s.setSession);
  return useMutation<SessionSnapshot, Error, string>({
    mutationFn: (sid: string) => sessionsApi.replayStart(sid),
    onSuccess: (snapshot: SessionSnapshot) => {
      qc.setQueryData(sessionKey(snapshot.session_id), snapshot);
      setSession(snapshot.session_id);
      window.history.pushState(null, "", `/s/${snapshot.session_id}`);
    },
  });
}

/** Step the replay forward by one section. The 3-second unified
 * loader is rendered in :class:`Rating.tsx`; this hook only owns the
 * network round-trip and cache update. */
export function useReplayAdvance(sid: string) {
  const qc = useQueryClient();
  return useMutation<SessionSnapshot, Error, void>({
    mutationFn: () => sessionsApi.replayAdvance(sid),
    onSuccess: (snapshot: SessionSnapshot) => {
      qc.setQueryData(sessionKey(snapshot.session_id), snapshot);
      // Refine + Apply summary endpoints serve from in-memory
      // ``state.refine_state`` / ``state.apply_state``; advancing
      // through replay attaches them one step at a time, so the
      // matching queries must refetch.
      qc.invalidateQueries({ queryKey: ["refine-summary", sid] });
      qc.invalidateQueries({ queryKey: ["apply-summary", sid] });
    },
  });
}
