import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { authApi, type AuthMeResponse } from "../api/session";

const meKey = ["auth", "me"];

/** Probe whether the current cookie carries a valid session. The
 * App calls this on mount to decide between rendering ``Login`` or
 * the rest of the tree. ``retry: false`` so a 401 lands as an error
 * promptly; the caller switches to ``<Login>`` based on ``isError``.
 */
export function useMe() {
  return useQuery<AuthMeResponse>({
    queryKey: meKey,
    queryFn: () => authApi.me(),
    retry: false,
    staleTime: Infinity,
  });
}

export function useLogin() {
  const qc = useQueryClient();
  return useMutation<AuthMeResponse, Error, string>({
    mutationFn: (token: string) => authApi.login(token),
    onSuccess: (res) => {
      qc.setQueryData<AuthMeResponse>(meKey, res);
    },
  });
}

export function useLogout() {
  const qc = useQueryClient();
  return useMutation<void, Error, void>({
    mutationFn: () => authApi.logout(),
    onSuccess: () => {
      // Drop everything scoped from the cache so a subsequent
      // login as another user doesn't see the previous user's data.
      qc.clear();
    },
  });
}
