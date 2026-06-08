import { create } from "zustand";
import type {
  ApplyOverrides,
  HarvestOverrides,
  RefineOverrides,
  SearchOverrides,
  SessionOverrides,
} from "../types";

export type Phase = "search" | "harvest" | "refine" | "apply";

type ParamOverridesState = {
  // The diff. A field is present here only when it differs from the
  // YAML default — empty objects/undefined mean "use default". The
  // diff is what we POST in the StartSession body.
  search: SearchOverrides | null;
  harvest: HarvestOverrides | null;
  refine: RefineOverrides | null;
  apply: ApplyOverrides | null;
} & {
  setPhase: <P extends Phase>(
    phase: P,
    block:
      | (P extends "search"
          ? SearchOverrides
          : P extends "harvest"
            ? HarvestOverrides
            : P extends "refine"
              ? RefineOverrides
              : ApplyOverrides)
      | null,
  ) => void;
  resetAll: () => void;
  buildSubmission: () => SessionOverrides | undefined;
  diffCount: () => number;
};

function isEmptyBlock(block: object | null): boolean {
  return !block || Object.keys(block).length === 0;
}

function nonEmpty<T extends object>(block: T | null): T | null {
  if (isEmptyBlock(block)) return null;
  return block;
}

export const useParamOverrides = create<ParamOverridesState>((set, get) => ({
  search: null,
  harvest: null,
  refine: null,
  apply: null,
  setPhase: (phase, block) => {
    // Coerce empty objects to null so the diff count stays accurate.
    const next = nonEmpty(block as object) as
      | SearchOverrides
      | HarvestOverrides
      | RefineOverrides
      | ApplyOverrides
      | null;
    set({ [phase]: next } as Partial<ParamOverridesState>);
  },
  resetAll: () => set({ search: null, harvest: null, refine: null, apply: null }),
  buildSubmission: () => {
    const { search, harvest, refine, apply } = get();
    const out: SessionOverrides = {};
    if (search) out.search = search;
    if (harvest) out.harvest = harvest;
    if (refine) out.refine = refine;
    if (apply) out.apply = apply;
    return Object.keys(out).length === 0 ? undefined : out;
  },
  diffCount: () => {
    const { search, harvest, refine, apply } = get();
    let n = 0;
    for (const block of [search, harvest, refine, apply]) {
      if (block) n += Object.keys(block).length;
    }
    return n;
  },
}));
