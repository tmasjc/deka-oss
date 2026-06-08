import { useState } from "react";
import { downloadBlob } from "../api/client";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";
import { RubricPromptModal } from "./RubricPromptModal";
import type { RefineSummary as RefineSummaryT } from "../types";

type Props = {
  sid: string;
  summary: RefineSummaryT;
  onEnd: () => void;
  // Frontend-only navigation back into the verdict browser in
  // read-only mode. Optional so legacy callers still compile.
  onRevisit?: () => void;
  // Plumb the Phase 4 hand-off — when ``apply.enabled`` the parent
  // screen wires this to kick off training. Omitted when apply is off.
  onApply?: () => void;
};

/**
 * Mirrors :class:`src.tui.widgets.refine_screens.RefineSummaryScreen`:
 * the shipped rubric version in the title, a one-line verdict count
 * with auto-drops broken out, an Estimated Total Chunks projection,
 * the per-decile distribution table (decile · distance range · n ·
 * Keep/Drop · Keep %), and the four sidecar paths.
 */
export function RefineSummary({
  sid,
  summary,
  onEnd,
  onRevisit,
  onApply,
}: Props) {
  const [promptOpen, setPromptOpen] = useState(false);

  useKeyboardShortcuts(
    {
      "ctrl+l": () =>
        void downloadBlob(
          `/api/session/${sid}/logs/download`,
          `session-${sid.slice(0, 8)}.zip`,
        ),
      q: () => {
        if (!promptOpen) onEnd();
      },
      enter: () => {
        if (promptOpen) return;
        if (onApply) onApply();
        else onEnd();
      },
      b: () => {
        if (!promptOpen) onRevisit?.();
      },
      v: () => setPromptOpen(true),
    },
    true,
  );

  return (
    <section className="refine-summary">
      <h2 className="modal__title">
        Phase 3 shipped — rubric v{summary.rubric_version}
      </h2>

      <div className="refine-summary__counts">
        Verdicts:{" "}
        <strong>KEEP={summary.keep_count}</strong>{" "}
        <strong>DROP={summary.drop_count}</strong>{" "}
        <strong>ERROR={summary.error_count}</strong>{" "}
        <strong>auto={summary.auto_drop_count}</strong>
      </div>
      <div className="refine-summary__estimate">
        Estimated Total Chunks:{" "}
        <strong>{summary.estimated_total_chunks}</strong>
      </div>

      <div className="modal__section-label">
        Sample distribution by FIT-distance decile
      </div>
      <table className="kv-table">
        <thead>
          <tr>
            <th>Decile</th>
            <th>Distance range</th>
            <th style={{ textAlign: "right" }}>n</th>
            <th>Keep / Drop</th>
            <th style={{ textAlign: "right" }}>Keep %</th>
          </tr>
        </thead>
        <tbody>
          {summary.decile_rows.map((row) => (
            <tr key={row.decile}>
              <th>{row.decile}</th>
              <td>{formatRange(row.distance_min, row.distance_max)}</td>
              <td style={{ textAlign: "right" }}>{row.sample_n}</td>
              <td>
                {row.keep_rate == null
                  ? "—"
                  : `${row.keep_count}/${row.drop_count}`}
              </td>
              <td style={{ textAlign: "right" }}>
                {row.keep_rate == null
                  ? "—"
                  : `${(row.keep_rate * 100).toFixed(1)}`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="modal__section-label">Sidecars</div>
      <div className="harvest-summary__sidecars">
        {(["prompt", "rubric", "evidence", "meta"] as const).map((key) => {
          const path = summary.sidecar_paths[key];
          if (!path) return null;
          return (
            <div key={key}>
              <strong>{capitalize(key)}:</strong> <code>{path}</code>
            </div>
          );
        })}
      </div>

      <div className="cfg__actions">
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
          onClick={() => setPromptOpen(true)}
        >
          <span className="btn__cap">View rubric prompt</span>
          <span className="btn__key">[v]</span>
        </button>
        {onRevisit && (
          <button type="button" className="btn" onClick={onRevisit}>
            <span className="btn__cap">Revisit verdicts</span>
            <span className="btn__key">[b]</span>
          </button>
        )}
        {onApply && (
          <button
            type="button"
            className="btn btn--fit"
            onClick={onApply}
            style={{ marginLeft: "auto" }}
          >
            <span className="btn__cap">Train classifier</span>
            <span className="btn__key">[↵]</span>
          </button>
        )}
        <button
          type="button"
          className="btn"
          onClick={onEnd}
          style={onApply ? undefined : { marginLeft: "auto" }}
        >
          <span className="btn__cap">End session</span>
          <span className="btn__key">[q]</span>
        </button>
      </div>

      {promptOpen && (
        <RubricPromptModal sid={sid} onClose={() => setPromptOpen(false)} />
      )}
    </section>
  );
}

function formatRange(lo: number | null, hi: number | null): string {
  if (lo == null || hi == null) return "—";
  return `${lo.toFixed(3)}–${hi.toFixed(3)}`;
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
