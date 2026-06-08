import { http } from "./client";
import type { Scope } from "../types";

export type ScopesResponse = { scopes: Scope[] };

export const scopesApi = {
  list: () => http.get<ScopesResponse>("/api/scopes"),
};
