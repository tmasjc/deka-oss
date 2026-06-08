import { useQuery } from "@tanstack/react-query";
import { configApi } from "../api/session";
import type { ConfigDefaults } from "../types";

const KEY = ["config", "defaults"] as const;

/**
 * Fetch the curated-essentials projection of the server's config.yaml.
 *
 * Used to pre-fill placeholders in the [Edit parameters] modal and to
 * compute "this field differs from default" for the override badge.
 * The defaults rarely change within a session, so we cache forever and
 * rely on a page refresh to pick up YAML edits.
 */
export function useConfigDefaults() {
  return useQuery<ConfigDefaults>({
    queryKey: KEY,
    queryFn: () => configApi.defaults(),
    staleTime: Infinity,
  });
}
