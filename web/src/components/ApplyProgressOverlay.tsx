import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useProgress } from "../hooks/useProgress";
import { ElapsedTimer } from "./ElapsedTimer";
import { LoadingHelix } from "./LoadingHelix";

type ApplyOverlayPhase =
  | "APPLY_TRAINING"
  | "APPLY_PREPARING"
  | "APPLY_APPLYING";

type Props = { sid: string; phase: ApplyOverlayPhase };

const COPY: Record<ApplyOverlayPhase, { title: string; detail: string }> = {
  APPLY_TRAINING: {
    title: "Training classifier",
    detail:
      "Fitting logistic regression on Phase 3 sample (BGE-M3 + nearest-fit distance).",
  },
  APPLY_PREPARING: {
    title: "Preparing calibration panel",
    detail:
      "Scoring the full cohort so the panel opens with the projection ready.",
  },
  APPLY_APPLYING: {
    title: "Shipping classifier",
    detail: "Writing KEEP / DROP verdicts across the cohort at the chosen τ.",
  },
};

/**
 * Full-screen overlay shown while a Phase 4 daemon worker runs. Same
 * shape as :class:`RefineProgressOverlay` / :class:`HarvestProgressOverlay`
 * so Phase 4 looks like a peer phase rather than an in-panel detour.
 */
export function ApplyProgressOverlay({ sid, phase }: Props) {
  const qc = useQueryClient();
  const progressQuery = useProgress(sid, true);
  const data = progressQuery.data;
  const stage = data?.stage;
  const error = data?.error ?? null;

  useEffect(() => {
    if (stage === "done" || error) {
      qc.invalidateQueries({ queryKey: ["session", sid] });
    }
  }, [stage, error, sid, qc]);

  const copy = COPY[phase];
  // Live detail from the worker if any; otherwise the static
  // per-phase blurb. Mirrors RefineProgressOverlay.
  const detail = data?.detail ?? copy.detail;

  return (
    <div className="turn-overlay" role="status" aria-live="polite">
      <LoadingHelix />
      <p className="turn-overlay__text">{copy.title}</p>
      <ElapsedTimer />
      <p className="turn-overlay__detail">{detail}</p>
      {error && <p className="modal__error">{error}</p>}
    </div>
  );
}
