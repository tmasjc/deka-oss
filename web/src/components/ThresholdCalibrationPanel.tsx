import { useEffect, useMemo, useRef, useState } from "react";
import { formatApiError } from "../api/client";
import {
  useApplyCancel,
  useApplyEval,
  useApplyFinalize,
} from "../hooks/useApply";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";
import { ShipConfirmModal } from "./ShipConfirmModal";
import type {
  ApplyEvalReport,
  ApplySummary,
  CohortProjection,
  PRCurvePoint,
} from "../types";

type Props = {
  sid: string;
  defaultThreshold: number;
  onFinalized: (summary: ApplySummary) => void;
  onCancelled?: () => void;
};

// Debounce live /apply/eval round-trips while the operator scrubs τ.
const DEBOUNCE_MS = 80;

function quantize(v: number): number {
  return Math.max(0, Math.min(1, Math.round(v * 100) / 100));
}

type Suggestion = { label: string; tau: number; precision: number; recall: number; f1: number };

function deriveSuggestions(pr: PRCurvePoint[]): Suggestion[] {
  if (pr.length === 0) return [];
  let best: PRCurvePoint = pr[0];
  let bestF1 = -Infinity;
  for (const p of pr) {
    const denom = p.precision + p.recall;
    const f1 = denom > 0 ? (2 * p.precision * p.recall) / denom : 0;
    if (f1 > bestF1) {
      bestF1 = f1;
      best = p;
    }
  }
  const p90 = pr.find((p) => p.precision >= 0.9) ?? pr[pr.length - 1];
  const p95 = pr.find((p) => p.precision >= 0.95) ?? pr[pr.length - 1];
  const f1Of = (p: PRCurvePoint) => {
    const d = p.precision + p.recall;
    return d > 0 ? (2 * p.precision * p.recall) / d : 0;
  };
  return [
    { label: "max F1", tau: quantize(best.threshold), precision: best.precision, recall: best.recall, f1: bestF1 },
    { label: "P ≥ 0.90", tau: quantize(p90.threshold), precision: p90.precision, recall: p90.recall, f1: f1Of(p90) },
    { label: "P ≥ 0.95", tau: quantize(p95.threshold), precision: p95.precision, recall: p95.recall, f1: f1Of(p95) },
  ];
}

type ConfusionCounts = { tp: number; fp: number; fn: number; tn: number };

function confusionFromAggregate(report: ApplyEvalReport): ConfusionCounts | null {
  // Reverse precision/recall/keep_n to TP/FP/FN/TN. Holds whenever the
  // backend's metrics are self-consistent at the evaluated τ.
  const { precision_at_threshold: p, recall_at_threshold: r, eval_keep_n, eval_drop_n } = report;
  if (eval_keep_n <= 0 || eval_drop_n < 0) return null;
  const tp = Math.round(r * eval_keep_n);
  const fn = eval_keep_n - tp;
  // precision = TP / (TP + FP). When precision==0, sklearn returns 0
  // either when TP==0 with FP>0 or when both are zero — we can't
  // distinguish without extra signal, so leave FP at 0 in that case.
  const fp = p > 0 ? Math.max(0, Math.round(tp / p) - tp) : 0;
  const tn = Math.max(0, eval_drop_n - fp);
  return { tp, fp, fn, tn };
}

function confusionFromEvalArrays(
  scores: number[] | undefined,
  labels: number[] | undefined,
  threshold: number,
): ConfusionCounts {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  const s = scores ?? [];
  const l = labels ?? [];
  for (let i = 0; i < s.length; i++) {
    const keep = s[i] >= threshold;
    const positive = l[i] === 1;
    if (keep && positive) tp++;
    else if (keep && !positive) fp++;
    else if (!keep && positive) fn++;
    else tn++;
  }
  return { tp, fp, fn, tn };
}

// Cohort histogram fallback: when eval set is empty (resume from disk),
// we still need *something* under τ. We use the borderline samples as a
// best-effort sketch — they're scored cohort PKs, unlabelled.
const HIST_BINS = 25;

function buildEvalHist(scores: number[] | undefined, labels: number[] | undefined) {
  const pos = new Array(HIST_BINS).fill(0);
  const neg = new Array(HIST_BINS).fill(0);
  const s = scores ?? [];
  const l = labels ?? [];
  for (let i = 0; i < s.length; i++) {
    const idx = Math.min(HIST_BINS - 1, Math.max(0, Math.floor(s[i] * HIST_BINS)));
    if (l[i] === 1) pos[idx]++;
    else neg[idx]++;
  }
  return { pos, neg, bins: HIST_BINS };
}

export function ThresholdCalibrationPanel({
  sid,
  defaultThreshold,
  onFinalized,
  onCancelled,
}: Props) {
  const [tau, setTauRaw] = useState<number>(quantize(defaultThreshold));
  const [debounced, setDebounced] = useState<number>(tau);
  // The "applied" τ is the threshold the operator commits via Apply. In
  // this single-shot flow there is only one apply event, so this stays
  // pinned to the initial default — the dirty / delta indicators
  // compare against the snapshot taken when the panel first mounted.
  const [appliedTau] = useState<number>(quantize(defaultThreshold));
  const [appliedSnapshot, setAppliedSnapshot] = useState<{
    metrics: { tp: number; fp: number; fn: number; tn: number };
    projection: CohortProjection | null;
  } | null>(null);
  const [hoverTau, setHoverTau] = useState<number | null>(null);
  const [allowLowPrecision, setAllowLowPrecision] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setTau = (v: number) => setTauRaw(quantize(v));

  useEffect(() => {
    const t = window.setTimeout(() => setDebounced(tau), DEBOUNCE_MS);
    return () => window.clearTimeout(t);
  }, [tau]);

  const evalQuery = useApplyEval(sid, true, debounced);
  const finalize = useApplyFinalize(sid);
  const cancel = useApplyCancel(sid);

  const projection = evalQuery.data?.projection ?? null;
  const evalAt: ApplyEvalReport | undefined = evalQuery.data?.eval_at_threshold;

  // Seed the applied snapshot from the first successful eval. Until then
  // the matrix and cohort show no delta chips — there's nothing to
  // compare against.
  useEffect(() => {
    if (appliedSnapshot === null && evalAt && projection) {
      const metrics =
        evalAt.eval_scores && evalAt.eval_scores.length > 0
          ? confusionFromEvalArrays(
              evalAt.eval_scores,
              evalAt.eval_labels,
              appliedTau,
            )
          : confusionFromAggregate(evalAt);
      if (metrics) setAppliedSnapshot({ metrics, projection });
    }
  }, [appliedSnapshot, evalAt, projection, appliedTau]);

  const passesBar = evalAt ? evalAt.passes_bar : true;
  const canFinalize =
    !!evalAt && !finalize.isPending && !cancel.isPending && (passesBar || allowLowPrecision);

  const suggestions = useMemo(
    () => deriveSuggestions(evalAt?.pr_curve ?? []),
    [evalAt?.pr_curve],
  );

  // Two paths to a confusion matrix:
  //   1. If the backend shipped raw eval_scores + eval_labels, recount
  //      from the vectors at the live τ (most accurate, lets each
  //      slider drag re-derive without a round-trip).
  //   2. Otherwise derive from the aggregate (precision, recall,
  //      eval_keep_n, eval_drop_n) at the τ the backend evaluated.
  //      Less responsive — only refreshes when the next /apply/eval
  //      returns — but works for resume-from-disk sessions and
  //      backends predating the raw-vector payload extension.
  const confusion = useMemo<ConfusionCounts | null>(() => {
    if (!evalAt) return null;
    if (evalAt.eval_scores && evalAt.eval_scores.length > 0) {
      return confusionFromEvalArrays(
        evalAt.eval_scores,
        evalAt.eval_labels,
        tau,
      );
    }
    return confusionFromAggregate(evalAt);
  }, [evalAt, tau]);

  const evalHist = useMemo(() => {
    if (!evalAt || !evalAt.eval_scores || evalAt.eval_scores.length === 0)
      return null;
    return buildEvalHist(evalAt.eval_scores, evalAt.eval_labels);
  }, [evalAt]);

  const dirty = Math.abs(tau - appliedTau) > 1e-6;

  // Pre-Ship confirmation gate. The bar's [Ship] click opens this; the
  // modal's confirm fires the actual /apply/finalize mutation.
  const [shipConfirmOpen, setShipConfirmOpen] = useState(false);

  function onShipClicked() {
    if (canFinalize) setShipConfirmOpen(true);
  }

  function onShipConfirmed() {
    setShipConfirmOpen(false);
    setError(null);
    finalize.mutate(
      { threshold: tau, allow_low_precision: !passesBar && allowLowPrecision },
      {
        onSuccess: (summary) => onFinalized(summary),
        onError: (err) => setError(formatApiError(err) ?? "Finalize failed."),
      },
    );
  }

  function onCancel() {
    setError(null);
    cancel.mutate(undefined, {
      onSuccess: () => onCancelled?.(),
      onError: (err) => setError(formatApiError(err) ?? "Cancel failed."),
    });
  }

  // Each panel-side handler is gated on ``!shipConfirmOpen`` so that
  // while the modal is up, keystrokes route only to the modal's
  // listeners. Without this, Esc would close the modal AND revert τ /
  // exit Phase 4 in the same tick; arrows would still scrub τ behind
  // the modal.
  useKeyboardShortcuts(
    {
      enter: () => {
        if (!shipConfirmOpen && canFinalize) onShipClicked();
      },
      escape: () => {
        if (shipConfirmOpen) return;
        if (dirty) setTau(appliedTau);
        else if (!finalize.isPending && !cancel.isPending) onCancel();
      },
      arrowleft: (e) => {
        if (!shipConfirmOpen) setTau(tau - (e.shiftKey ? 0.05 : 0.01));
      },
      arrowright: (e) => {
        if (!shipConfirmOpen) setTau(tau + (e.shiftKey ? 0.05 : 0.01));
      },
    },
    true,
  );

  const minPrecision = evalAt?.min_precision ?? 0.9;

  return (
    <div className="calibration">
      <Stats
        tau={tau}
        evalAt={evalAt}
        minPrecision={minPrecision}
        passesBar={passesBar}
        dirty={dirty}
      />

      <div className="calibration__grid-top">
        <ThresholdControl
          tau={tau}
          appliedTau={appliedTau}
          setTau={setTau}
          suggestions={suggestions}
        />
        <CohortCard projection={projection} appliedProjection={appliedSnapshot?.projection ?? null} />
      </div>

      <div className="calibration__grid-bot">
        <Histogram
          hist={evalHist}
          tau={tau}
          setTau={setTau}
          suggestions={suggestions}
          hoverTau={hoverTau}
          setHoverTau={setHoverTau}
        />
        <PRCurvePanel
          pr={evalAt?.pr_curve ?? []}
          tau={tau}
          setTau={setTau}
          suggestions={suggestions}
          minPrecision={minPrecision}
        />
        <ConfusionMatrixPanel
          counts={confusion}
          previous={appliedSnapshot?.metrics ?? null}
          evalN={evalAt?.eval_n ?? 0}
        />
      </div>

      {!passesBar && (
        <div className="calibration__warning">
          <strong>Eval precision below bar.</strong>{" "}
          The current threshold gives{" "}
          {evalAt?.precision_at_threshold.toFixed(3)}, below the configured
          minimum of {minPrecision.toFixed(2)}. Raise τ to be more
          conservative, or explicitly accept the risk:
          <label>
            <input
              type="checkbox"
              checked={allowLowPrecision}
              onChange={(e) => setAllowLowPrecision(e.target.checked)}
            />{" "}
            Allow shipping below the precision bar (records{" "}
            <code>operator_decision: "override_low_precision"</code>).
          </label>
        </div>
      )}

      {error && <div className="calibration__error">{error}</div>}

      <ApplyBar
        tau={tau}
        appliedTau={appliedTau}
        dirty={dirty}
        canFinalize={canFinalize}
        finalizing={finalize.isPending}
        cancelling={cancel.isPending}
        cohortTotal={projection?.total ?? null}
        onApply={onShipClicked}
        onCancel={onCancel}
      />
      {shipConfirmOpen && (
        <ShipConfirmModal
          onClose={() => setShipConfirmOpen(false)}
          onConfirmed={onShipConfirmed}
        />
      )}
    </div>
  );
}

// ─── Stats strip ────────────────────────────────────────────────────────────

function Stats({
  tau,
  evalAt,
  minPrecision,
  passesBar,
  dirty,
}: {
  tau: number;
  evalAt: ApplyEvalReport | undefined;
  minPrecision: number;
  passesBar: boolean;
  dirty: boolean;
}) {
  const precision = evalAt?.precision_at_threshold ?? null;
  const recall = evalAt?.recall_at_threshold ?? null;
  const f1 =
    precision !== null && recall !== null && precision + recall > 0
      ? (2 * precision * recall) / (precision + recall)
      : null;
  return (
    <div className="calibration__stats">
      <StatCell label="τ" className="cal-stat--tau">
        <span className="cal-stat__value">{tau.toFixed(2)}</span>
        {dirty && <span className="cal-stat__dirty">unsaved</span>}
      </StatCell>
      <StatCell label="precision" className={passesBar ? "cal-stat--good" : "cal-stat--bad"}>
        <span className="cal-stat__value">{precision !== null ? precision.toFixed(3) : "—"}</span>
        <span className={`cal-stat__minchip ${passesBar ? "cal-stat__minchip--ok" : "cal-stat__minchip--fail"}`}>
          min {minPrecision.toFixed(2)}
        </span>
      </StatCell>
      <StatCell label="recall">
        <span className="cal-stat__value">{recall !== null ? recall.toFixed(3) : "—"}</span>
      </StatCell>
      <StatCell label="F1">
        <span className="cal-stat__value">{f1 !== null ? f1.toFixed(3) : "—"}</span>
      </StatCell>
      <StatCell label="eval n">
        <span className="cal-stat__value">{evalAt?.eval_n ?? 0}</span>
        {evalAt && (
          <span className="cal-stat__breakdown">
            <span className="cal-stat__keep">{evalAt.eval_keep_n} keep</span>
            <span className="cal-stat__sep">·</span>
            <span className="cal-stat__drop">{evalAt.eval_drop_n} drop</span>
          </span>
        )}
      </StatCell>
    </div>
  );
}

function StatCell({
  label,
  className,
  children,
}: {
  label: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={`cal-stat ${className ?? ""}`}>
      <div className="cal-stat__label">{label}</div>
      <div className="cal-stat__row">{children}</div>
    </div>
  );
}

// ─── Threshold control ──────────────────────────────────────────────────────

function ThresholdControl({
  tau,
  appliedTau,
  setTau,
  suggestions,
}: {
  tau: number;
  appliedTau: number;
  setTau: (v: number) => void;
  suggestions: Suggestion[];
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);

  const tauFromEvent = (e: React.PointerEvent) => {
    if (!trackRef.current) return tau;
    const rect = trackRef.current.getBoundingClientRect();
    return quantize((e.clientX - rect.left) / rect.width);
  };

  return (
    <div className="cal-card cal-thr">
      <div className="cal-thr__label">
        <span>CONFIDENCE THRESHOLD</span>
        <span className="cal-thr__sub">τ ∈ [0, 1] · step 0.01 · ← / → keys</span>
      </div>
      <div className="cal-thr__row">
        <div
          ref={trackRef}
          className={`cal-thr__track ${dragging ? "cal-thr__track--drag" : ""}`}
          onPointerDown={(e) => {
            setDragging(true);
            trackRef.current?.setPointerCapture(e.pointerId);
            setTau(tauFromEvent(e));
          }}
          onPointerMove={(e) => {
            if (dragging) setTau(tauFromEvent(e));
          }}
          onPointerUp={(e) => {
            setDragging(false);
            try {
              trackRef.current?.releasePointerCapture(e.pointerId);
            } catch {
              /* noop */
            }
          }}
        >
          <div className="cal-thr__fill" style={{ width: `${tau * 100}%` }} />
          <div
            className="cal-thr__applied"
            style={{ left: `${appliedTau * 100}%` }}
            title={`applied at τ = ${appliedTau.toFixed(2)}`}
          />
          {suggestions.map((s) => (
            <div
              key={s.label}
              className="cal-thr__sug"
              style={{ left: `${s.tau * 100}%` }}
              title={`${s.label} · τ = ${s.tau.toFixed(2)}`}
              onPointerDown={(e) => {
                e.stopPropagation();
                setTau(s.tau);
              }}
            />
          ))}
          <div className="cal-thr__knob" style={{ left: `${tau * 100}%` }} />
        </div>
        <div className="cal-thr__input">
          <input
            type="text"
            value={tau.toFixed(2)}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              if (!Number.isNaN(v)) setTau(v);
            }}
            aria-label="threshold value"
          />
        </div>
      </div>
      {suggestions.length > 0 && (
        <div className="cal-thr__sug-row">
          {suggestions.map((s) => (
            <button
              key={s.label}
              type="button"
              className={`cal-thr__chip ${Math.abs(tau - s.tau) < 0.005 ? "cal-thr__chip--on" : ""}`}
              onClick={() => setTau(s.tau)}
              title={`P=${s.precision.toFixed(3)} · R=${s.recall.toFixed(3)} · F1=${s.f1.toFixed(3)}`}
            >
              <span className="cal-thr__chip-dot" />
              <span className="cal-thr__chip-label">{s.label}</span>
              <span className="cal-thr__chip-tau">τ {s.tau.toFixed(2)}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Cohort projection card ─────────────────────────────────────────────────

function CohortCard({
  projection,
  appliedProjection,
}: {
  projection: CohortProjection | null;
  appliedProjection: CohortProjection | null;
}) {
  if (!projection) {
    return (
      <div className="cal-card cal-cohort">
        <div className="cal-card__head">
          <div className="cal-card__title">PROJECTED COHORT</div>
        </div>
        <div className="cal-cohort__empty">Loading projection…</div>
      </div>
    );
  }
  const keepPct = projection.total > 0 ? projection.keep / projection.total : 0;
  const dropPct = projection.total > 0 ? projection.drop / projection.total : 0;
  const deltaKeep =
    appliedProjection !== null ? projection.keep - appliedProjection.keep : 0;
  return (
    <div className="cal-card cal-cohort">
      <div className="cal-card__head">
        <div className="cal-card__title">
          PROJECTED COHORT @ τ = {projection.threshold.toFixed(2)}
        </div>
        <div className="cal-card__legend">{projection.total.toLocaleString()} total</div>
      </div>
      <div className="cal-cohort__stack">
        <div className="cal-cohort__bar">
          <div className="cal-cohort__keep" style={{ width: `${keepPct * 100}%` }} />
          <div className="cal-cohort__drop" style={{ width: `${dropPct * 100}%` }} />
        </div>
        <div className="cal-cohort__nums">
          <div className="cal-cohort__cell cal-cohort__cell--keep">
            <div className="cal-cohort__n">{projection.keep.toLocaleString()}</div>
            <div className="cal-cohort__l">KEEP · {(keepPct * 100).toFixed(1)}%</div>
          </div>
          <div className="cal-cohort__cell cal-cohort__cell--drop">
            <div className="cal-cohort__n">{projection.drop.toLocaleString()}</div>
            <div className="cal-cohort__l">DROP · {(dropPct * 100).toFixed(1)}%</div>
          </div>
          {deltaKeep !== 0 && (
            <div className="cal-cohort__cell cal-cohort__cell--delta">
              <div className="cal-cohort__n">
                {deltaKeep > 0 ? "+" : ""}
                {deltaKeep.toLocaleString()}
              </div>
              <div className="cal-cohort__l">Δ KEEP vs applied</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Score-distribution histogram ───────────────────────────────────────────

function niceTicks(max: number, count: number): number[] {
  if (max <= 0) return [0];
  const step = Math.pow(10, Math.floor(Math.log10(max / count)));
  const m = max / count / step;
  const niceM = m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10;
  const s = niceM * step;
  const out: number[] = [];
  for (let v = 0; v <= max; v += s) out.push(Math.round(v));
  return out;
}

function Histogram({
  hist,
  tau,
  setTau,
  suggestions,
  hoverTau,
  setHoverTau,
}: {
  hist: { pos: number[]; neg: number[]; bins: number } | null;
  tau: number;
  setTau: (v: number) => void;
  suggestions: Suggestion[];
  hoverTau: number | null;
  setHoverTau: (v: number | null) => void;
}) {
  const W = 700;
  const H = 280;
  const PAD_L = 44;
  const PAD_R = 18;
  const PAD_T = 24;
  const PAD_B = 32;
  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;

  const svgRef = useRef<SVGSVGElement>(null);
  const dragging = useRef(false);

  const xFor = (s: number) => PAD_L + s * innerW;
  const tauX = xFor(tau);
  const hoverX = hoverTau !== null ? xFor(hoverTau) : null;

  const tauFromEvent = (e: React.PointerEvent) => {
    if (!svgRef.current) return tau;
    const rect = svgRef.current.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    const s = (ratio * W - PAD_L) / innerW;
    return quantize(s);
  };

  const onPointerDown = (e: React.PointerEvent) => {
    dragging.current = true;
    svgRef.current?.setPointerCapture(e.pointerId);
    setTau(tauFromEvent(e));
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const t = tauFromEvent(e);
    setHoverTau(t);
    if (dragging.current) setTau(t);
  };
  const onPointerUp = (e: React.PointerEvent) => {
    dragging.current = false;
    try {
      svgRef.current?.releasePointerCapture(e.pointerId);
    } catch {
      /* noop */
    }
  };
  const onPointerLeave = () => setHoverTau(null);

  const ticks: number[] = [];
  for (let i = 0; i <= 10; i++) ticks.push(i / 10);

  const maxCount = useMemo(() => {
    if (!hist) return 0;
    let m = 0;
    for (let i = 0; i < hist.bins; i++) m = Math.max(m, hist.pos[i] + hist.neg[i]);
    return m;
  }, [hist]);

  const yTicks = useMemo(() => niceTicks(maxCount, 4), [maxCount]);

  const sumArr = (a: number[]) => a.reduce((s, x) => s + x, 0);
  const posTotal = hist ? sumArr(hist.pos) : 0;
  const negTotal = hist ? sumArr(hist.neg) : 0;

  return (
    <div className="cal-card cal-hist">
      <div className="cal-card__head">
        <div className="cal-card__title">SCORE DISTRIBUTION</div>
        <div className="cal-card__legend">
          <span className="cal-leg cal-leg--pos">
            <span className="cal-leg__sw" />
            positive ({posTotal.toLocaleString()})
          </span>
          <span className="cal-leg cal-leg--neg">
            <span className="cal-leg__sw" />
            negative ({negTotal.toLocaleString()})
          </span>
          <span className="cal-leg cal-leg--muted">drag τ to retune</span>
        </div>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="cal-hist__svg"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerLeave}
      >
        {yTicks.map((t, i) => {
          const y = maxCount > 0 ? PAD_T + innerH - (t / maxCount) * innerH : PAD_T + innerH;
          return (
            <g key={`y${i}`}>
              <line x1={PAD_L} x2={W - PAD_R} y1={y} y2={y} className="cal-grid-y" />
              <text x={PAD_L - 8} y={y + 4} className="cal-ax-label cal-ax-label--r">
                {t.toLocaleString()}
              </text>
            </g>
          );
        })}

        {ticks.map((t, i) => {
          const x = xFor(t);
          return (
            <g key={`x${i}`}>
              <line x1={x} x2={x} y1={PAD_T + innerH} y2={PAD_T + innerH + 5} className="cal-tick" />
              <text x={x} y={PAD_T + innerH + 18} className="cal-ax-label cal-ax-label--c">
                {t.toFixed(1)}
              </text>
            </g>
          );
        })}
        <text x={(PAD_L + W - PAD_R) / 2} y={H - 4} className="cal-ax-title">
          predicted score →
        </text>
        <text
          x={-(PAD_T + innerH / 2)}
          y={14}
          transform="rotate(-90)"
          className="cal-ax-title"
        >
          count
        </text>

        <rect
          x={PAD_L}
          y={PAD_T}
          width={Math.max(0, tauX - PAD_L)}
          height={innerH}
          className="cal-region cal-region--drop"
        />
        <rect
          x={tauX}
          y={PAD_T}
          width={Math.max(0, W - PAD_R - tauX)}
          height={innerH}
          className="cal-region cal-region--keep"
        />

        {hist && maxCount > 0 && (
          <>
            {Array.from({ length: hist.bins }).map((_, i) => {
              const score = (i + 0.5) / hist.bins;
              const keep = score >= tau;
              const barW = innerW / hist.bins;
              const posH = (hist.pos[i] / maxCount) * innerH;
              const negH = (hist.neg[i] / maxCount) * innerH;
              const x = PAD_L + i * barW;
              const baseY = PAD_T + innerH;
              const negCls = keep ? "cal-bar--fp" : "cal-bar--tn";
              const posCls = keep ? "cal-bar--tp" : "cal-bar--fn";
              return (
                <g key={i}>
                  <rect
                    x={x + 0.6}
                    y={baseY - negH}
                    width={Math.max(0.4, barW - 1.2)}
                    height={negH}
                    className={`cal-bar ${negCls}`}
                  />
                  <rect
                    x={x + 0.6}
                    y={baseY - negH - posH}
                    width={Math.max(0.4, barW - 1.2)}
                    height={posH}
                    className={`cal-bar ${posCls}`}
                  />
                </g>
              );
            })}
          </>
        )}

        {suggestions.map((s) => {
          const x = xFor(s.tau);
          return (
            <g key={s.label} className="cal-sugg">
              <line x1={x} x2={x} y1={PAD_T} y2={PAD_T + innerH} className="cal-sugg__line" />
              <rect x={x - 38} y={PAD_T - 18} width={76} height={14} rx={2} className="cal-sugg__pill" />
              <text x={x} y={PAD_T - 7} className="cal-sugg__label">
                {s.label}
              </text>
            </g>
          );
        })}

        {hoverX !== null && Math.abs(hoverX - tauX) > 1 && (
          <line
            x1={hoverX}
            x2={hoverX}
            y1={PAD_T}
            y2={PAD_T + innerH}
            className="cal-hover-line"
          />
        )}

        <line
          x1={tauX}
          x2={tauX}
          y1={PAD_T - 4}
          y2={PAD_T + innerH + 4}
          className="cal-tau-line"
        />
        <polygon
          points={`${tauX},${PAD_T - 6} ${tauX - 6},${PAD_T - 14} ${tauX + 6},${PAD_T - 14}`}
          className="cal-tau-handle"
        />
        <polygon
          points={`${tauX},${PAD_T + innerH + 6} ${tauX - 6},${PAD_T + innerH + 14} ${tauX + 6},${PAD_T + innerH + 14}`}
          className="cal-tau-handle"
        />
      </svg>
      <div className="cal-hist__foot">
        <span className="cal-hist__foot-drop">← drop (score &lt; τ)</span>
        <span className="cal-hist__foot-keep">keep (score ≥ τ) →</span>
      </div>
    </div>
  );
}

// ─── PR curve panel ─────────────────────────────────────────────────────────

function PRCurvePanel({
  pr,
  tau,
  setTau,
  suggestions,
  minPrecision,
}: {
  pr: PRCurvePoint[];
  tau: number;
  setTau: (v: number) => void;
  suggestions: Suggestion[];
  minPrecision: number;
}) {
  const W = 460;
  const H = 280;
  const PAD_L = 40;
  const PAD_R = 16;
  const PAD_T = 18;
  const PAD_B = 32;
  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;
  const xFor = (r: number) => PAD_L + r * innerW;
  const yFor = (p: number) => PAD_T + innerH - p * innerH;

  const path = useMemo(() => {
    if (pr.length === 0) return "";
    const sorted = [...pr].sort((a, b) => a.recall - b.recall);
    return (
      "M " +
      sorted
        .map((p) => `${xFor(p.recall).toFixed(2)},${yFor(p.precision).toFixed(2)}`)
        .join(" L ")
    );
  }, [pr]);

  const isoF1 = [0.5, 0.6, 0.7, 0.8, 0.9];
  const isoPath = (f1: number) => {
    const pts: string[] = [];
    for (let r = f1 / 2 + 0.005; r <= 1; r += 0.01) {
      const p = (f1 * r) / (2 * r - f1);
      if (p >= 0 && p <= 1) pts.push(`${xFor(r).toFixed(2)},${yFor(p).toFixed(2)}`);
    }
    return pts.length ? "M " + pts.join(" L ") : "";
  };

  const cur = useMemo(() => {
    if (pr.length === 0) return null;
    let nearest = pr[0];
    let best = Math.abs(pr[0].threshold - tau);
    for (const p of pr) {
      const d = Math.abs(p.threshold - tau);
      if (d < best) {
        best = d;
        nearest = p;
      }
    }
    return nearest;
  }, [pr, tau]);

  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverPt, setHoverPt] = useState<PRCurvePoint | null>(null);

  const findNearest = (e: React.PointerEvent): PRCurvePoint | null => {
    if (!svgRef.current) return null;
    const rect = svgRef.current.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const py = ((e.clientY - rect.top) / rect.height) * H;
    let nearest: PRCurvePoint | null = null;
    let best = Infinity;
    for (const p of pr) {
      const dx = xFor(p.recall) - px;
      const dy = yFor(p.precision) - py;
      const d = dx * dx + dy * dy;
      if (d < best) {
        best = d;
        nearest = p;
      }
    }
    return Math.sqrt(best) < 28 ? nearest : null;
  };

  const xTicks = [0, 0.25, 0.5, 0.75, 1];
  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="cal-card cal-pr">
      <div className="cal-card__head">
        <div className="cal-card__title">PR CURVE</div>
        <div className="cal-card__legend">
          <span className="cal-leg cal-leg--muted">iso-F1</span>
          <span className="cal-leg cal-leg--curve">
            <span className="cal-leg__sw" />
            precision @ τ
          </span>
        </div>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="cal-pr__svg"
        onPointerMove={(e) => setHoverPt(findNearest(e))}
        onPointerLeave={() => setHoverPt(null)}
        onClick={(e) => {
          const n = findNearest(e as unknown as React.PointerEvent);
          if (n) setTau(quantize(n.threshold));
        }}
      >
        <rect
          x={PAD_L}
          y={PAD_T}
          width={innerW}
          height={yFor(minPrecision) - PAD_T}
          className="cal-pr__minband"
        />
        <line
          x1={PAD_L}
          x2={PAD_L + innerW}
          y1={yFor(minPrecision)}
          y2={yFor(minPrecision)}
          className="cal-pr__minline"
        />
        <text x={PAD_L + 6} y={yFor(minPrecision) - 4} className="cal-pr__minlabel">
          min precision {minPrecision.toFixed(2)}
        </text>

        {isoF1.map((f1) => (
          <g key={f1}>
            <path d={isoPath(f1)} className="cal-iso" />
            <text
              x={xFor(0.99)}
              y={yFor((f1 * 0.99) / (2 * 0.99 - f1)) - 2}
              className="cal-iso__label"
              textAnchor="end"
            >
              F1 {f1.toFixed(1)}
            </text>
          </g>
        ))}

        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line
              x1={xFor(t)}
              x2={xFor(t)}
              y1={PAD_T + innerH}
              y2={PAD_T + innerH + 5}
              className="cal-tick"
            />
            <text x={xFor(t)} y={PAD_T + innerH + 18} className="cal-ax-label cal-ax-label--c">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line
              x1={PAD_L - 5}
              x2={PAD_L}
              y1={yFor(t)}
              y2={yFor(t)}
              className="cal-tick"
            />
            <text x={PAD_L - 8} y={yFor(t) + 4} className="cal-ax-label cal-ax-label--r">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        <text x={PAD_L + innerW / 2} y={H - 4} className="cal-ax-title">
          recall →
        </text>
        <text
          x={-(PAD_T + innerH / 2)}
          y={14}
          transform="rotate(-90)"
          className="cal-ax-title"
        >
          precision ↑
        </text>

        <rect x={PAD_L} y={PAD_T} width={innerW} height={innerH} className="cal-pr__frame" />

        {pr.length > 1 && <path d={path} className="cal-pr__curve" />}

        {suggestions.map((s) => {
          const pt = pr.find((p) => Math.abs(p.threshold - s.tau) < 0.006);
          if (!pt) return null;
          return (
            <g key={s.label} transform={`translate(${xFor(pt.recall)}, ${yFor(pt.precision)})`}>
              <circle r={5} className="cal-pr__sugout" />
              <circle r={2.5} className="cal-pr__sugin" />
            </g>
          );
        })}

        {hoverPt && (
          <g transform={`translate(${xFor(hoverPt.recall)}, ${yFor(hoverPt.precision)})`}>
            <circle r={9} className="cal-pr__hover" />
            <g transform="translate(10, -8)">
              <rect x={0} y={-12} width={148} height={32} rx={3} className="cal-pr__tipbg" />
              <text x={8} y={1} className="cal-pr__tipline1">
                τ {hoverPt.threshold.toFixed(2)} · click to set
              </text>
              <text x={8} y={15} className="cal-pr__tipline2">
                P {hoverPt.precision.toFixed(3)} · R {hoverPt.recall.toFixed(3)}
              </text>
            </g>
          </g>
        )}

        {cur && (
          <g transform={`translate(${xFor(cur.recall)}, ${yFor(cur.precision)})`}>
            <circle r={9} className="cal-pr__curring" />
            <circle r={4.5} className="cal-pr__curdot" />
          </g>
        )}
      </svg>
    </div>
  );
}

// ─── Confusion matrix ───────────────────────────────────────────────────────

function ConfusionMatrixPanel({
  counts,
  previous,
  evalN,
}: {
  counts: ConfusionCounts | null;
  previous: ConfusionCounts | null;
  evalN: number;
}) {
  const renderDelta = (a: number | undefined, b: number | undefined) => {
    if (a === undefined || b === undefined) return null;
    const d = a - b;
    if (d === 0) return null;
    return (
      <span className={`cal-cm__delta ${d > 0 ? "cal-cm__delta--up" : "cal-cm__delta--down"}`}>
        {d > 0 ? "+" : ""}
        {d}
      </span>
    );
  };
  return (
    <div className="cal-card cal-cm">
      <div className="cal-card__head">
        <div className="cal-card__title">CONFUSION MATRIX</div>
        <div className="cal-card__legend">
          <span className="cal-leg cal-leg--muted">eval n = {evalN}</span>
        </div>
      </div>
      <div className="cal-cm__grid">
        <div className="cal-cm__corner" />
        <div className="cal-cm__h">pred. KEEP</div>
        <div className="cal-cm__h">pred. DROP</div>

        <div className="cal-cm__h cal-cm__h--r">actual +</div>
        <div className="cal-cm__cell cal-cm__cell--good">
          <div className="cal-cm__lbl">TP</div>
          <div className="cal-cm__val">
            {counts ? counts.tp : "—"} {renderDelta(counts?.tp, previous?.tp)}
          </div>
        </div>
        <div className="cal-cm__cell cal-cm__cell--bad">
          <div className="cal-cm__lbl">FN</div>
          <div className="cal-cm__val">
            {counts ? counts.fn : "—"} {renderDelta(counts?.fn, previous?.fn)}
          </div>
        </div>

        <div className="cal-cm__h cal-cm__h--r">actual −</div>
        <div className="cal-cm__cell cal-cm__cell--bad">
          <div className="cal-cm__lbl">FP</div>
          <div className="cal-cm__val">
            {counts ? counts.fp : "—"} {renderDelta(counts?.fp, previous?.fp)}
          </div>
        </div>
        <div className="cal-cm__cell cal-cm__cell--good">
          <div className="cal-cm__lbl">TN</div>
          <div className="cal-cm__val">
            {counts ? counts.tn : "—"} {renderDelta(counts?.tn, previous?.tn)}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Apply bar (in-panel footer) ────────────────────────────────────────────

function ApplyBar({
  tau,
  appliedTau,
  dirty,
  canFinalize,
  finalizing,
  cancelling,
  cohortTotal,
  onApply,
  onCancel,
}: {
  tau: number;
  appliedTau: number;
  dirty: boolean;
  canFinalize: boolean;
  finalizing: boolean;
  cancelling: boolean;
  cohortTotal: number | null;
  onApply: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="cal-applybar">
      <div className="cal-applybar__left">
        <button
          type="button"
          className="cal-btn"
          onClick={(e) => {
            onCancel();
            e.currentTarget.blur();
          }}
          disabled={finalizing || cancelling}
        >
          <span>{cancelling ? "Cancelling…" : "Cancel"}</span>
          <kbd>esc</kbd>
        </button>
        {dirty && (
          <span className="cal-applybar__note">
            <span className="cal-applybar__dot" /> τ moved from{" "}
            <b>{appliedTau.toFixed(2)}</b> → <b>{tau.toFixed(2)}</b>
          </span>
        )}
      </div>
      <div className="cal-applybar__right">
        {cohortTotal !== null && (
          <span className="cal-applybar__hint">
            shipping will retag the {cohortTotal.toLocaleString()}-row cohort
          </span>
        )}
        <button
          type="button"
          className={`cal-btn cal-btn--primary ${dirty ? "" : "cal-btn--ghost"}`}
          onClick={(e) => {
            onApply();
            e.currentTarget.blur();
          }}
          disabled={!canFinalize}
        >
          <span>{finalizing ? "Shipping…" : `Ship at τ ${tau.toFixed(2)}`}</span>
          <kbd>↵</kbd>
        </button>
      </div>
    </div>
  );
}
