import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useProgress } from "../hooks/useProgress";
import { ElapsedTimer } from "./ElapsedTimer";
import { LoadingHelix } from "./LoadingHelix";

type Props = { sid: string };

/**
 * Full-screen overlay shown while the harvest worker runs. The runner
 * (``src/anchor/runner.py``) emits free-form progress strings; we
 * surface the most recent line as the overlay's detail message — same
 * shape the TUI's ``AnchorProgressScreen`` renders.
 *
 * The harvest worker runs in a daemon thread on the backend; there's
 * no mutation response to refresh the React Query cache when it
 * finishes. We watch the ``/progress`` stream and invalidate the
 * session query the moment ``stage`` becomes ``done`` (or an error
 * surfaces) so the snapshot re-fetches and the overlay's parent
 * unmounts it on phase transition.
 */
export function HarvestProgressOverlay({ sid }: Props) {
  const qc = useQueryClient();
  const progressQuery = useProgress(sid, true);
  const stage = progressQuery.data?.stage;
  const error = progressQuery.data?.error ?? null;

  useEffect(() => {
    if (stage === "done" || error) {
      qc.invalidateQueries({ queryKey: ["session", sid] });
    }
  }, [stage, error, sid, qc]);

  const detail = progressQuery.data?.detail ?? "Loading FITs from session…";

  return (
    <div className="turn-overlay" role="status" aria-live="polite">
      <LoadingHelix />
      <p className="turn-overlay__text">FIT-anchored retrieval running</p>
      <ElapsedTimer />
      <p className="turn-overlay__detail">{detail}</p>
      {error && <p className="modal__error">{error}</p>}
    </div>
  );
}
