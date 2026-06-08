import { useQuery } from "@tanstack/react-query";
import { sessionApi, type ProgressDTO } from "../api/session";

/**
 * Polls /api/session/{sid}/progress while `active` is true.
 *
 * Caller toggles `active` to match the lifetime of the overlay so polling
 * stops the moment the blocking mutation resolves. The poll interval is
 * short enough to feel live (500 ms) but long enough to stay well under
 * the per-chunk extraction rate (~300 ms).
 */
export function useProgress(sid: string | null, active: boolean) {
  return useQuery<ProgressDTO>({
    queryKey: ["progress", sid ?? ""],
    queryFn: () => sessionApi.progress(sid!),
    enabled: !!sid && active,
    refetchInterval: active ? 500 : false,
    refetchIntervalInBackground: false,
    staleTime: 0,
    gcTime: 0,
    retry: false,
  });
}
