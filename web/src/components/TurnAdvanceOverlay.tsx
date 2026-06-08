import { useEffect, useRef, useState } from "react";
import type { ProgressDTO } from "../api/session";
import { LoadingHelix } from "./LoadingHelix";

export type OverlayPhase = "pending" | "flash";

export type OverlayDiagnostics = {
  fetched: number | null;
  precision: number | null;
  observe: string | null;
  converged: boolean;
};

export type Stage = { key: string; label: string; expectedMs: number };

// Backend stage keys are authoritative once /progress responds. The
// time-based `expectedMs` budgets below are only used as a fallback for
// the first ~500 ms before the first poll resolves.
export const ADVANCE_STAGES: Stage[] = [
  { key: "reflecting", label: "Reflecting on turn", expectedMs: 4000 },
  { key: "adapting", label: "Adapting search config", expectedMs: 500 },
  { key: "searching_milvus", label: "Searching Milvus", expectedMs: 3000 },
  { key: "extracting_spans", label: "Extracting spans", expectedMs: 30000 },
];

export const START_STAGES: Stage[] = [
  { key: "probing", label: "Probing retrieval paths", expectedMs: 2500 },
  { key: "adapting", label: "Adapting search config", expectedMs: 500 },
  { key: "searching_milvus", label: "Searching Milvus", expectedMs: 3000 },
  { key: "extracting_spans", label: "Extracting spans", expectedMs: 30000 },
];

type Props = {
  phase: OverlayPhase;
  pendingTitle: string;
  flashTitle?: string;
  stages?: Stage[];
  diagnostics?: OverlayDiagnostics | null;
  progress?: ProgressDTO | null;
};

export function TurnAdvanceOverlay({
  phase,
  pendingTitle,
  flashTitle,
  stages = ADVANCE_STAGES,
  diagnostics = null,
  progress = null,
}: Props) {
  const startedAt = useRef<number>(Date.now());
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    if (phase !== "pending") return;
    const prefersReducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const tick = prefersReducedMotion ? 500 : 100;
    const id = window.setInterval(() => {
      setElapsedMs(Date.now() - startedAt.current);
    }, tick);
    return () => window.clearInterval(id);
  }, [phase]);

  if (phase === "flash") {
    return (
      <FlashView title={flashTitle ?? pendingTitle} diagnostics={diagnostics} />
    );
  }

  const liveStageIndex =
    progress && progress.stage !== "idle"
      ? stages.findIndex((s) => s.key === progress.stage)
      : -1;
  const usingLive = liveStageIndex >= 0;
  const totalExpectedMs = stages.reduce((acc, s) => acc + s.expectedMs, 0);
  const activeStageIndex = usingLive
    ? liveStageIndex
    : stageIndexFor(elapsedMs, stages);
  const overTime = !usingLive && elapsedMs >= totalExpectedMs;

  // Backend may surface a free-form per-attempt note (e.g. auto-retry
  // bumping per_path_limit). Render it under the elapsed counter so the
  // operator understands a longer wait. The note is also stamped onto
  // the active stage's label as a fallback when the stage matches.
  const detail = progress?.detail ?? null;

  return (
    <div className="turn-overlay" role="status" aria-live="polite">
      <LoadingHelix />
      <p className="turn-overlay__text">{pendingTitle}</p>
      <p className="turn-overlay__elapsed">{formatElapsed(elapsedMs)}</p>
      {detail && <p className="turn-overlay__detail">{detail}</p>}
      <ul className="turn-overlay__stages">
        {stages.map((stage, i) => {
          const state =
            i < activeStageIndex
              ? "done"
              : i === activeStageIndex
                ? "active"
                : "pending";
          const counter =
            state === "active" &&
            usingLive &&
            progress &&
            stage.key === "extracting_spans" &&
            progress.total != null
              ? ` — ${progress.processed} / ${progress.total} chunks`
              : "";
          const suffix =
            state === "active" && overTime
              ? " — taking longer than usual"
              : "";
          return (
            <li
              key={stage.key}
              className={`turn-overlay__stage turn-overlay__stage--${state}`}
            >
              <StageIcon state={state} />
              <span className="turn-overlay__stage-label">
                {stage.label}
                {counter}
                {suffix}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function FlashView({
  title,
  diagnostics,
}: {
  title: string;
  diagnostics: OverlayDiagnostics | null;
}) {
  return (
    <div className="turn-overlay" role="status" aria-live="polite">
      <div className="turn-overlay__flash">
        <div className="turn-overlay__flash-heading">
          <StageIcon state="done" />
          <span>{title}</span>
        </div>
        {diagnostics && (
          <dl className="turn-overlay__flash-list">
            {diagnostics.fetched != null && (
              <div className="turn-overlay__flash-row">
                <dt>Candidates</dt>
                <dd>{diagnostics.fetched}</dd>
              </div>
            )}
            {diagnostics.precision != null && (
              <div className="turn-overlay__flash-row">
                <dt>Precision</dt>
                <dd>{diagnostics.precision.toFixed(2)}</dd>
              </div>
            )}
            {diagnostics.converged && (
              <div className="turn-overlay__flash-row">
                <dt>Status</dt>
                <dd>Session converged</dd>
              </div>
            )}
            {diagnostics.observe && (
              <div className="turn-overlay__flash-observe">
                {diagnostics.observe}
              </div>
            )}
          </dl>
        )}
      </div>
    </div>
  );
}

function StageIcon({ state }: { state: "done" | "active" | "pending" }) {
  if (state === "done") {
    return (
      <svg
        className="turn-overlay__stage-icon turn-overlay__stage-icon--done"
        viewBox="0 0 16 16"
        aria-hidden="true"
      >
        <path
          d="M3.5 8.5 L6.5 11.5 L12.5 5"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }
  if (state === "active") {
    return <span className="turn-overlay__stage-icon turn-overlay__stage-icon--active" />;
  }
  return <span className="turn-overlay__stage-icon turn-overlay__stage-icon--pending" />;
}

function stageIndexFor(elapsedMs: number, stages: Stage[]): number {
  let acc = 0;
  for (let i = 0; i < stages.length; i++) {
    acc += stages[i].expectedMs;
    if (elapsedMs < acc) return i;
  }
  return stages.length - 1;
}

function formatElapsed(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}
