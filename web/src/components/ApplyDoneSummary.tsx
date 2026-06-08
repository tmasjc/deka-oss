import { useEffect, useState } from "react";
import { downloadBlob } from "../api/client";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";
import { RubricPromptModal } from "./RubricPromptModal";
import { TextScramble } from "./TextScramble";
import type { ApplySummary } from "../types";

type Props = {
  sid: string;
  summary: ApplySummary;
  // True on fresh-completion first view; false on DONE_VIEW resume so
  // the body renders immediately without the scramble theatre.
  playScramble: boolean;
  // True when the session is read-only (DONE_VIEW / APPLY_PENDING
  // resume). Footer button copy and modal copy adapt.
  readOnly: boolean;
  // Called when the operator wants to end (writable) or close
  // (read-only). Rating.tsx gates the actual mutation behind a
  // confirmation modal — this handler just opens it.
  onEnd: () => void;
};

/**
 * Terminal Phase 4 summary card. On fresh completion the headline
 * scrambles in and the body fades; on DONE_VIEW resume the screen
 * renders flat (the user has seen the reveal before).
 *
 * Layout reads top→bottom: hero stat → query → secondary grid →
 * session time → collapsed sidecars → footer actions.
 */
export function ApplyDoneSummary({
  sid,
  summary,
  playScramble,
  readOnly,
  onEnd,
}: Props) {
  const [scrambleDone, setScrambleDone] = useState(!playScramble);
  const [rubricOpen, setRubricOpen] = useState(false);

  // If the parent flips ``playScramble`` from false → true after mount,
  // re-arm the animation. The normal lifecycle is mount-with-true, so
  // this only fires on the unusual case of an in-place reveal-gate
  // dismissal that re-renders this component.
  useEffect(() => {
    if (playScramble) setScrambleDone(false);
  }, [playScramble]);

  // Keep shortcuts dormant while the reveal sequence is in progress so
  // that an Enter press confirming the reveal-gate modal does not also
  // fire ``onEnd`` here in the same event tick. ``useKeyboardShortcuts``
  // is a plain global listener with no priority, so the gate is
  // enforced at this caller.
  useKeyboardShortcuts(
    {
      "ctrl+l": () =>
        void downloadBlob(
          `/api/session/${sid}/logs/download`,
          `session-${sid.slice(0, 8)}.zip`,
        ),
      q: () => onEnd(),
      enter: () => onEnd(),
    },
    scrambleDone && !rubricOpen,
  );

  const proj = summary.cohort_projection;
  const ev = summary.eval;
  const sidecars = summary.sidecar_paths;
  const elapsed = formatElapsed(
    summary.session_started_at,
    summary.session_ended_at,
  );

  return (
    <section className="apply-summary">
      <h2 className="apply-summary__headline">
        {playScramble && !scrambleDone ? (
          <TextScramble
            text="Cohort finalized"
            durationMs={1400}
            onDone={() => setScrambleDone(true)}
          />
        ) : (
          "Cohort finalized"
        )}
      </h2>

      <div
        className="apply-summary__body"
        style={{
          opacity: scrambleDone ? 1 : 0,
          transition: "opacity 400ms ease-out, transform 400ms ease-out",
          transform: scrambleDone ? "translateY(0)" : "translateY(6px)",
        }}
        aria-hidden={!scrambleDone}
      >
        <div className="apply-summary__hero">
          <div className="apply-summary__hero-stat">
            {proj.keep.toLocaleString()}
            <span className="apply-summary__hero-unit"> kept</span>
          </div>
          <div className="apply-summary__hero-sub">
            of {proj.total.toLocaleString()} candidates ·{" "}
            <strong>{(ev.precision_at_threshold * 100).toFixed(1)}%</strong>{" "}
            precision ·{" "}
            <strong>{(ev.recall_at_threshold * 100).toFixed(1)}%</strong>{" "}
            recall
          </div>
        </div>

        {summary.query && (
          <div className="apply-summary__query">
            <div className="apply-summary__label">Query</div>
            <div className="apply-summary__query-text">{summary.query}</div>
          </div>
        )}

        {/*
          Row pairings deliberate (not alphabetical):
          - Threshold ↔ Operator decision: τ picked + the operator's call on it
          - Training N ↔ Eval N: sample sizes used to fit + measure
          - Min-precision bar ↔ Passed bar: expectation vs outcome
          - Rubric version ↔ Dropped: housekeeping / audit fields
          Precision + recall live in the hero strip above; not repeated here.
        */}
        <div className="apply-summary__grid">
          <KvRow label="Threshold (τ)" value={summary.threshold.toFixed(3)} />
          <KvRow label="Operator decision" value={summary.operator_decision} />
          <KvRow label="Training N" value={summary.training_n.toLocaleString()} />
          <KvRow label="Eval N" value={ev.eval_n.toLocaleString()} />
          <KvRow
            label="Min-precision bar"
            value={ev.min_precision.toFixed(2)}
          />
          <KvRow label="Passed bar" value={ev.passes_bar ? "yes" : "no"} />
          <KvRow
            label="Rubric version"
            value={`v${summary.rubric_version}`}
          />
          <KvRow label="Dropped" value={proj.drop.toLocaleString()} />
        </div>

        {elapsed && (
          <div className="apply-summary__time">
            <span className="apply-summary__label">Session time</span>
            <strong>{elapsed}</strong>
          </div>
        )}

        <details className="apply-summary__sidecars">
          <summary>Sidecar files ({Object.keys(sidecars).length})</summary>
          <div className="apply-summary__sidecars-body">
            {(["classifier", "eval", "labels", "meta"] as const).map((key) => {
              const path = sidecars[key];
              if (!path) return null;
              return (
                <div key={key}>
                  <strong>{capitalize(key)}:</strong> <code>{path}</code>
                </div>
              );
            })}
          </div>
        </details>

        <div className="apply-summary__footer">
          <button
            type="button"
            className="btn"
            onClick={() =>
              downloadBlob(
                `/api/session/${sid}/logs/download`,
                `session-${sid.slice(0, 8)}.zip`,
              )
            }
          >
            <span className="btn__cap">Download logs</span>
            <span className="btn__key">[ctrl+l]</span>
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => setRubricOpen(true)}
          >
            <span className="btn__cap">View rubric</span>
          </button>
          <button
            type="button"
            className="btn btn--fit"
            onClick={onEnd}
            style={{ marginLeft: "auto" }}
          >
            <span className="btn__cap">
              {readOnly ? "Close" : "End session"}
            </span>
            <span className="btn__key">[q]</span>
          </button>
        </div>
      </div>
      {rubricOpen && (
        <RubricPromptModal sid={sid} onClose={() => setRubricOpen(false)} />
      )}
    </section>
  );
}

function KvRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="apply-summary__kv">
      <span className="apply-summary__kv-label">{label}</span>
      <span className="apply-summary__kv-value">{value}</span>
    </div>
  );
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function formatElapsed(start: string | null, end: string | null): string | null {
  if (!start || !end) return null;
  const t0 = Date.parse(start);
  const t1 = Date.parse(end);
  if (Number.isNaN(t0) || Number.isNaN(t1) || t1 < t0) return null;
  const seconds = Math.round((t1 - t0) / 1000);
  if (seconds < 60) return `${seconds}s`;
  const mins = Math.floor(seconds / 60);
  const rem = seconds % 60;
  if (mins < 60) return `${mins}m ${rem.toString().padStart(2, "0")}s`;
  const hours = Math.floor(mins / 60);
  const remMins = mins % 60;
  return `${hours}h ${remMins.toString().padStart(2, "0")}m ${rem.toString().padStart(2, "0")}s`;
}
