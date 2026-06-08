import { create } from "zustand";
import { ApiError, formatApiError } from "../api/client";
import { sessionApi } from "../api/session";

export type LeftTab = "turn" | "pk" | "paths" | "config";

type UiState = {
  sessionId: string | null;
  cursor: number;
  reflectionOpen: boolean;
  configOpen: boolean;
  activeTab: LeftTab;
  banner: { kind: "info" | "error"; text: string } | null;
  filtersBannerDismissed: Set<number>;
  probeBannerDismissed: boolean;
  recommendationDismissed: Set<number>;
  expandedPks: Set<string>;
  originalCache: Map<string, string>;
  expandingPks: Set<string>;
  setSession: (id: string | null) => void;
  setCursor: (n: number) => void;
  bumpCursor: () => void;
  resetCursor: () => void;
  setReflectionOpen: (v: boolean) => void;
  setConfigOpen: (v: boolean) => void;
  setActiveTab: (tab: LeftTab) => void;
  setBanner: (b: UiState["banner"]) => void;
  dismissFilterBanner: (turn: number) => void;
  dismissProbeBanner: () => void;
  dismissRecommendation: (turn: number) => void;
  toggleExpanded: (pk: number | string, sid: string) => Promise<void>;
};

function resetExpansion() {
  return {
    expandedPks: new Set<string>(),
    originalCache: new Map<string, string>(),
    expandingPks: new Set<string>(),
  };
}

export const useUi = create<UiState>((set, get) => ({
  sessionId: null,
  cursor: 0,
  reflectionOpen: false,
  configOpen: false,
  activeTab: "turn",
  banner: null,
  filtersBannerDismissed: new Set<number>(),
  probeBannerDismissed: false,
  recommendationDismissed: new Set<number>(),
  ...resetExpansion(),
  setSession: (id) =>
    set({
      sessionId: id,
      cursor: 0,
      filtersBannerDismissed: new Set(),
      probeBannerDismissed: false,
      recommendationDismissed: new Set(),
      ...resetExpansion(),
    }),
  setCursor: (n) => set({ cursor: n }),
  bumpCursor: () => set((s) => ({ cursor: s.cursor + 1 })),
  resetCursor: () => set({ cursor: 0 }),
  setReflectionOpen: (v) => set({ reflectionOpen: v }),
  setConfigOpen: (v) => set({ configOpen: v }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  setBanner: (b) => set({ banner: b }),
  dismissFilterBanner: (turn) =>
    set((s) => ({
      filtersBannerDismissed: new Set([...s.filtersBannerDismissed, turn]),
    })),
  dismissProbeBanner: () => set({ probeBannerDismissed: true }),
  dismissRecommendation: (turn) =>
    set((s) => ({
      recommendationDismissed: new Set([...s.recommendationDismissed, turn]),
    })),
  toggleExpanded: async (pk, sid) => {
    const key = String(pk);
    const s = get();

    // Collapse path.
    if (s.expandedPks.has(key)) {
      const next = new Set(s.expandedPks);
      next.delete(key);
      set({ expandedPks: next });
      return;
    }

    // Expand path. If already cached, just flip state — no fetch.
    if (s.originalCache.has(key)) {
      set({ expandedPks: new Set([...s.expandedPks, key]) });
      return;
    }

    // Cache miss: fetch, mark loading so UI can indicate it.
    if (s.expandingPks.has(key)) return; // de-dupe concurrent presses
    set({
      expandedPks: new Set([...s.expandedPks, key]),
      expandingPks: new Set([...s.expandingPks, key]),
    });
    try {
      const res = await sessionApi.fetchOriginal(sid, pk);
      set((cur) => {
        const cache = new Map(cur.originalCache);
        cache.set(key, res.original_content);
        const expanding = new Set(cur.expandingPks);
        expanding.delete(key);
        return { originalCache: cache, expandingPks: expanding };
      });
    } catch (err) {
      set((cur) => {
        const expanded = new Set(cur.expandedPks);
        expanded.delete(key);
        const expanding = new Set(cur.expandingPks);
        expanding.delete(key);
        const text =
          err instanceof ApiError && err.status === 403
            ? "Context expansion has been disabled by the server admin."
            : err instanceof ApiError && err.status === 404
            ? "Original content not available for this chunk."
            : `Could not load original content — ${formatApiError(err)}`;
        return {
          expandedPks: expanded,
          expandingPks: expanding,
          banner: { kind: "error", text },
        };
      });
    }
  },
}));
