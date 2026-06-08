import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useProgress } from "../hooks/useProgress";
import { ElapsedTimer } from "./ElapsedTimer";
import { LoadingHelix } from "./LoadingHelix";

type Props = { sid: string; mode: "deriving" | "judging" };

/**
 * Refine workers run in daemon threads — same as harvest — so we
 * invalidate the session query when ``stage`` reports ``done`` (or
 * an error) so the parent un-mounts this overlay on phase transition.
 */
export function RefineProgressOverlay({ sid, mode }: Props) {
  const qc = useQueryClient();
  const progressQuery = useProgress(sid, true);
  const data = progressQuery.data;
  const error = data?.error ?? null;
  const stage = data?.stage;

  useEffect(() => {
    if (stage === "done" || error) {
      qc.invalidateQueries({ queryKey: ["session", sid] });
    }
  }, [stage, error, sid, qc]);

  const title = mode === "deriving" ? "Deriving rubric" : "Judging sample";
  const showBar = mode === "judging" && data?.total != null && data.total > 0;
  const fraction = showBar
    ? Math.min(1, (data!.processed ?? 0) / (data!.total as number))
    : 0;
  let detail: string;
  if (mode === "judging" && data?.total != null) {
    detail = `${data.processed} / ${data.total} verdicts`;
  } else {
    detail = data?.detail ?? "Working…";
  }

  return (
    <div className="turn-overlay" role="status" aria-live="polite">
      <LoadingHelix />
      <p className="turn-overlay__text">{title}</p>
      <ElapsedTimer />
      <p className="turn-overlay__detail">{detail}</p>
      {showBar && (
        <div
          className="progress-bar"
          role="progressbar"
          aria-valuenow={Math.round(fraction * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            className="progress-bar__fill"
            style={{ width: `${(fraction * 100).toFixed(1)}%` }}
          />
        </div>
      )}
      {error && <p className="modal__error">{error}</p>}
    </div>
  );
}
