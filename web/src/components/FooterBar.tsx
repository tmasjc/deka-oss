import type { Params } from "../types";

type Props = { params: Params; phase?: string | null };

type Key = { k: string; label: string; kind?: "pos" | "neg" };

const TUNING_KEYS: Key[] = [
  { k: "f", label: "FIT", kind: "pos" },
  { k: "n", label: "NOT_FIT", kind: "neg" },
  { k: "d", label: "Discard" },
  { k: "x", label: "Expand" },
  { k: "p", label: "Audit" },
  { k: "o", label: "Drop path" },
  { k: "e", label: "Edit config" },
  { k: "a", label: "Next turn" },
  { k: "r", label: "Reflection" },
  { k: "^l", label: "Download logs" },
  { k: "q", label: "Quit" },
];

const HARVEST_DONE_KEYS: Key[] = [
  { k: "↵", label: "Continue → Refine", kind: "pos" },
  { k: "^l", label: "Download logs" },
  { k: "q", label: "End session" },
];

const REFINE_EDITING_KEYS: Key[] = [
  { k: "^s", label: "Save rubric" },
  { k: "^↵", label: "Run judge", kind: "pos" },
  { k: "q", label: "Quit" },
];

const REFINE_REVIEW_KEYS: Key[] = [
  { k: "j", label: "Next" },
  { k: "k", label: "Prev" },
  { k: "d", label: "Discard", kind: "neg" },
  { k: "↵", label: "Finalize", kind: "pos" },
  { k: "q", label: "Quit" },
];

const DONE_KEYS: Key[] = [
  { k: "^l", label: "Download logs" },
  { k: "q", label: "End session" },
];

const APPLY_REVIEW_KEYS: Key[] = [
  { k: "←/→", label: "Adjust τ (±0.01)" },
  { k: "⇧←/→", label: "Adjust τ (±0.05)" },
  { k: "esc", label: "Cancel / revert", kind: "neg" },
  { k: "↵", label: "Ship at τ", kind: "pos" },
  { k: "q", label: "Quit" },
];

const QUIT_ONLY: Key[] = [{ k: "q", label: "Quit" }];

function keysFor(phase: string | null | undefined): Key[] {
  switch (phase) {
    case "ANCHOR_RUNNING":
    case "REFINE_DERIVING":
    case "REFINE_JUDGING":
    case "APPLY_TRAINING":
    case "APPLY_PREPARING":
    case "APPLY_APPLYING":
      return QUIT_ONLY;
    case "ANCHOR_DONE":
      return HARVEST_DONE_KEYS;
    case "ANCHOR_FAILED":
    case "REFINE_FAILED":
    case "APPLY_FAILED":
      return QUIT_ONLY;
    case "REFINE_EDITING":
      return REFINE_EDITING_KEYS;
    case "REFINE_REVIEW":
      return REFINE_REVIEW_KEYS;
    case "APPLY_REVIEW":
    case "APPLY_CONFIRM":
      return APPLY_REVIEW_KEYS;
    case "DONE":
      return DONE_KEYS;
    case "TUNING":
    default:
      return TUNING_KEYS;
  }
}

export function FooterBar({ params, phase }: Props) {
  const keys = keysFor(phase);
  // The "Current params" line is only meaningful while still tuning;
  // once the session has handed off to harvest/refine the params are
  // frozen and the footer space is better used for phase-relevant
  // status (rubric version etc. are surfaced in the header).
  const showParams = !phase || phase === "TUNING";
  const paramsLine = showParams
    ? `RRF k=${params.rrf_k}  limit=${params.per_path_limit}  top_k=${params.top_k}  paths=${params.active_paths.join(",")}`
    : "";
  return (
    <footer className="foot">
      <div className="foot__current">
        {showParams ? (
          <>
            <span className="foot__ctitle">Current</span>
            <span className="foot__params">{paramsLine}</span>
          </>
        ) : (
          <span className="foot__ctitle">{phaseLabel(phase)}</span>
        )}
      </div>
      <div className="foot__keys">
        {keys.map((k) => (
          <span key={k.k} className={"kc" + (k.kind ? " kc--" + k.kind : "")}>
            <span className="kc__k">{k.k}</span>
            <span className="kc__l">{k.label}</span>
          </span>
        ))}
      </div>
      <div className="foot__right">
        <span className="kc">
          <span className="kc__k">^p</span>
          <span className="kc__l">palette</span>
        </span>
      </div>
    </footer>
  );
}

function phaseLabel(phase: string | null | undefined): string {
  switch (phase) {
    case "ANCHOR_RUNNING":
      return "Phase 2 — harvesting";
    case "ANCHOR_DONE":
      return "Phase 2 — review summary";
    case "ANCHOR_FAILED":
      return "Phase 2 — failed";
    case "REFINE_DERIVING":
      return "Phase 3 — deriving rubric";
    case "REFINE_EDITING":
      return "Phase 3 — edit rubric";
    case "REFINE_JUDGING":
      return "Phase 3 — judging sample";
    case "REFINE_REVIEW":
      return "Phase 3 — review verdicts";
    case "REFINE_FAILED":
      return "Phase 3 — failed";
    case "APPLY_TRAINING":
      return "Phase 4 — training classifier";
    case "APPLY_PREPARING":
      return "Phase 4 — preparing calibration panel";
    case "APPLY_REVIEW":
      return "Phase 4 — calibrate threshold";
    case "APPLY_APPLYING":
      return "Phase 4 — shipping classifier";
    case "APPLY_CONFIRM":
      return "Phase 4 — confirm";
    case "APPLY_FAILED":
      return "Phase 4 — failed";
    case "DONE":
      return "Done";
    default:
      return "";
  }
}
