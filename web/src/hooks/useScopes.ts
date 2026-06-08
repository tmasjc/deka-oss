import { useQuery } from "@tanstack/react-query";
import { scopesApi } from "../api/scopes";

export function useScopes() {
  return useQuery({
    queryKey: ["scopes"],
    queryFn: () => scopesApi.list(),
    staleTime: Infinity,
    retry: 1,
  });
}
