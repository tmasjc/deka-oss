import { downloadBlob, formatApiError } from "../api/client";
import { useHarvestResult } from "../hooks/useHarvest";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";
import type { AnchorResult, FrequencyGate } from "../types";

type Props = {
  sid: string;
  onEnd: () => void;
  /** When set, renders a "Continue → Refine" button. Skipping it ends the
   * session at the converged Phase 1 + Phase 2 logs. */
  onContinue?: () => void;
};

export function HarvestSummary({ sid, onEnd, onContinue }: Props) {
  const q = useHarvestResult(sid, true);

  function downloadLogs() {
    void downloadBlob(
      `/api/session/${sid}/logs/download`,
      `session-${sid.slice(0, 8)}.zip`,
    );
  }

  useKeyboardShortcuts(
    {
      enter: () => {
        if (onContinue) onContinue();
      },
      "ctrl+l": downloadLogs,
      q: () => onEnd(),
    },
    true,
  );

  if (q.isLoading) {
    return <div className="loading">Loading harvest summary…</div>;
  }
  if (q.isError || !q.data) {
    return (
      <div className="loading">
        {formatApiError(q.error) ?? "Harvest result unavailable."}
      </div>
    );
  }

  const r = q.data;
  return (
    <section className="harvest-summary">
      <h2 className="modal__title">
        Phase 2 complete — verdict: <strong>{r.verdict}</strong>
      </h2>

      <SectionTable
        title="Calibration & retrieval"
        rows={[
          ["LOO recovery", `${r.loo_recovered}/${r.loo_total}`],
          [
            "T (pre→post-drop)",
            `${r.quality_gate_T_pre_drop.toFixed(4)} → ${r.T.toFixed(4)}`,
          ],
          [
            "δ (per-FIT range)",
            `${r.delta_min.toFixed(4)} – ${r.delta_median.toFixed(4)} – ${r.delta_max.toFixed(4)}`,
          ],
          [
            "T' (per-FIT range)",
            `${r.T_prime_min.toFixed(4)} – ${r.T_prime_median.toFixed(4)} – ${r.T_prime_max.toFixed(4)}`,
          ],
          [
            "Radius scheme",
            r.radius_scheme === "decoupled"
              ? `decoupled  T'_out=${r.T_prime_out.toFixed(4)}`
              : r.radius_scheme,
          ],
          ["Phase-1 negatives (auto-DROP)", String(r.not_fit_intrusions)],
        ]}
      />

      <QualityGateSection result={r} />

      {r.frequency_gate && (
        <FrequencyGateSection
          freq={r.frequency_gate}
          retained={r.retained_chunks}
          nDiscardFiltered={r.n_discard_filtered}
        />
      )}

      <Warnings result={r} />

      <div className="harvest-summary__sidecars">
        {r.sidecar_jsonl_path ? (
          <>
            <div>
              <strong>JSONL sidecar:</strong> <code>{r.sidecar_jsonl_path}</code>
            </div>
            <div>
              <strong>Meta sidecar:</strong> <code>{r.sidecar_meta_path}</code>
            </div>
          </>
        ) : (
          <div>(dry-run — no sidecars written)</div>
        )}
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
        <button type="button" className="btn" onClick={onEnd}>
          <span className="btn__cap">End session</span>
          <span className="btn__key">[q]</span>
        </button>
        {onContinue && (
          <button type="button" className="btn btn--fit" onClick={onContinue}>
            <span className="btn__cap">Continue → Refine</span>
            <span className="btn__key">[↵]</span>
          </button>
        )}
      </div>
    </section>
  );
}

function SectionTable({
  title,
  rows,
}: {
  title: string;
  rows: [string, string][];
}) {
  return (
    <div className="harvest-summary__section">
      <div className="modal__section-label">{title}</div>
      <table className="kv-table">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <th>{k}</th>
              <td>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function QualityGateSection({ result }: { result: AnchorResult }) {
  const entering = result.n_fit_entering_quality_gate;
  const drops = result.quality_gate_dropped;
  // Survivors = entering − dropped. Falls back to entering when the
  // entering count was never populated (older backends / dry-run).
  const surviving = entering > 0 ? entering - drops.length : entering;
  // Under issue #47's floored-median logic the multiplier rule is
  // always live, so ``cutoff`` is populated on every new run and
  // ``median_floor_applied`` distinguishes the regime. ``cutoff ===
  // null`` only appears on legacy sidecars from before #47 (and on
  // replay paths that don't yet hydrate the cutoff field).
  const cutoff = result.quality_gate_multiplier_cutoff;
  const median = result.quality_gate_median_delta_pre_drop;
  const floorApplied = result.quality_gate_median_floor_applied;
  let ruleRow: [string, string];
  if (cutoff === null) {
    // Legacy sidecar from before issue #47 — pre-floor logic could
    // disable the rule entirely when median(δ) sank below 1e-3.
    ruleRow = [
      "Multiplier rule",
      `disabled — median ${median.toFixed(4)} ≤ 1e-3 floor`,
    ];
  } else if (floorApplied) {
    ruleRow = [
      "Multiplier rule",
      `active (floor backstop) — cutoff ${cutoff.toFixed(4)} (${result.quality_gate_multiplier.toFixed(1)} × 0.005 floor)`,
    ];
  } else {
    ruleRow = [
      "Multiplier rule",
      `active — cutoff ${cutoff.toFixed(4)} (${result.quality_gate_multiplier.toFixed(1)} × median)`,
    ];
  }

  return (
    <div className="harvest-summary__section">
      <div className="modal__section-label">Quality gate</div>
      <table className="kv-table">
        <tbody>
          <tr>
            <th>Entering / surviving</th>
            <td>
              {entering} → {surviving}
            </td>
          </tr>
          <tr>
            <th>median(δ) pre-drop</th>
            <td>{result.quality_gate_median_delta_pre_drop.toFixed(4)}</td>
          </tr>
          <tr>
            <th>{ruleRow[0]}</th>
            <td>{ruleRow[1]}</td>
          </tr>
          {drops.map((d) => (
            <tr key={`qg-${d.fit_chunk_id}`}>
              <th>Dropped</th>
              <td>
                {d.fit_chunk_id} (δ={d.delta.toFixed(4)}) — {d.reasons.join(", ")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FrequencyGateSection({
  freq,
  retained,
  nDiscardFiltered,
}: {
  freq: FrequencyGate;
  retained: number;
  nDiscardFiltered: number;
}) {
  const histo = freq.qualifying_count_histogram ?? {};
  const f = freq.f_configured;
  const bucketEntries = Object.entries(histo)
    .map(([k, v]) => [Number(k), v] as [number, number])
    .sort((a, b) => a[0] - b[0]);
  const atF = bucketEntries
    .filter(([k]) => k === f)
    .reduce((acc, [, v]) => acc + v, 0);
  const aboveF = bucketEntries
    .filter(([k]) => k > f)
    .reduce((acc, [, v]) => acc + v, 0);
  const histoLabel = bucketEntries.length
    ? bucketEntries.map(([k, v]) => `${k}:${v}`).join(" ")
    : "—";

  const rows: [string, string][] = [
    ["f (configured)", String(f)],
    ["Candidates kept", String(freq.kept)],
  ];
  if (nDiscardFiltered > 0) {
    rows.push(["DISCARD pks filtered", String(nDiscardFiltered)]);
  }
  rows.push(["Retained chunks (final)", String(retained)]);
  rows.push([
    "Qualifying anchors per chunk",
    `at f=${f} / above f    ${atF.toLocaleString()} / ${aboveF.toLocaleString()}`,
  ]);
  rows.push(["histogram", histoLabel]);

  return <SectionTable title="Anchor-frequency gate" rows={rows} />;
}

function Warnings({ result }: { result: AnchorResult }) {
  const cm = result.cohort_consistency_missing;
  const ex = result.budget_exhausted;
  if (cm.length === 0 && ex.length === 0) return null;
  const isError = ex.length > 0;
  return (
    <div className={"harvest-summary__warnings" + (isError ? " is-error" : "")}>
      <div className="modal__section-label">Warnings</div>
      <table className="kv-table">
        <tbody>
          {cm.map((m) => (
            <tr key={`cm-${m.fit_chunk_id}`}>
              <th>Cohort consistency</th>
              <td>{m.fit_chunk_id} — own chunk absent</td>
            </tr>
          ))}
          {ex.map((id) => (
            <tr key={`ex-${id}`}>
              <th>Budget exhausted</th>
              <td>{id} — raise harvest.max_k</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
