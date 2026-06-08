import { useQuery } from "@tanstack/react-query";
import { sessionApi } from "../api/session";

const KEY = (sid: string) => ["session", sid, "overrides"] as const;

/**
 * Fetch the session's overrides sidecar (per-phase deltas from YAML).
 *
 * Empty dict for sessions that didn't set any overrides. Combined with
 * `useConfigDefaults` to render the effective config in the sidebar
 * phase panels.
 */
export function useSessionOverrides(sid: string) {
  return useQuery<Record<string, Record<string, unknown>>>({
    queryKey: KEY(sid),
    queryFn: () => sessionApi.overrides(sid),
    staleTime: Infinity,
  });
}
