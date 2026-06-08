import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { sessionApi } from "../api/session";
import type { AnchorResult, HarvestPreflight, SessionSnapshot } from "../types";

const sessionKey = (sid: string) => ["session", sid];
const harvestResultKey = (sid: string) => ["harvest-result", sid];

export function useHarvestPreflight(sid: string) {
  return useMutation({
    mutationFn: (): Promise<HarvestPreflight> =>
      sessionApi.harvestPreflight(sid),
  });
}

export function useHarvestRun(sid: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => sessionApi.harvestRun(sid),
    onSuccess: (snapshot: SessionSnapshot) => {
      qc.setQueryData(sessionKey(sid), snapshot);
    },
  });
}

/**
 * Fetches the harvest result once the worker reports ANCHOR_DONE. The
 * caller controls ``enabled`` (typically tied to ``snap.phase ===
 * "ANCHOR_DONE"``) so we don't 409/404 while the run is still in
 * flight or before the user kicked it off.
 */
export function useHarvestResult(sid: string, enabled: boolean) {
  return useQuery<AnchorResult>({
    queryKey: harvestResultKey(sid),
    queryFn: () => sessionApi.harvestResult(sid),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}
