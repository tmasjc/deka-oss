import { useEffect, useRef, useState } from "react";
import {
  type PreflightCheck,
  type PreflightFailureDetail,
  type PreflightResponse,
} from "../api/session";
import { TextScramble } from "../components/TextScramble";

type PreflightProps = {
  scope: string;
  onPass: () => void;
  onCancel: () => void;
};

type Phase = "running" | "revealing" | "all_passed" | "failed";

const REVEAL_INTERVAL_MS = 130;
// Operator-readable countdown after all checks turn green. Gives the
// reader a beat to register the all-pass before the screen yanks away.
const COUNTDOWN_START = 3;

/**
 * Pre-flight screen (issue #33).
 *
 * Renders a scrambled headline, then animates the check list in one
 * row at a time. On all-green it auto-advances by calling ``onPass``;
 * on any failure the reveal halts at the failing item and shows the
 * typed error (env_var + detail) with a Retry button that re-runs.
 */
export function Preflight({ scope, onPass, onCancel }: PreflightProps) {
  const [phase, setPhase] = useState<Phase>("running");
  const [checks, setChecks] = useState<PreflightCheck[]>([]);
  const [failure, setFailure] = useState<PreflightFailureDetail | null>(null);
  const [revealed, setRevealed] = useState(0);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [headlineDone, setHeadlineDone] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [countdown, setCountdown] = useState<number>(COUNTDOWN_START);
  // Latch the auto-advance call so a re-render doesn't double-trigger
  // the parent's onPass.
  const advancedRef = useRef(false);

  useEffect(() => {
    advancedRef.current = false;
    setPhase("running");
    setChecks([]);
    setFailure(null);
    setRevealed(0);
    setRequestError(null);
    setHeadlineDone(false);
    setCountdown(COUNTDOWN_START);

    let cancelled = false;
    // Direct fetch (not the http wrapper) so the structured 400
    // ``detail`` object survives — the wrapper only preserves
    // ``detail`` when it's a string and would otherwise discard
    // ``checks`` / ``code`` / ``env_var``.
    fetch("/api/session/preflight", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ scope: scope }),
    })
      .then(async (res) => {
        if (cancelled) return;
        const body = await res.json().catch(() => null);
        if (res.ok) {
          const ok = body as PreflightResponse;
          setChecks(ok.checks);
          setFailure(null);
          return;
        }
        if (res.status === 400 && body && body.detail) {
          const detail = body.detail as PreflightFailureDetail;
          if (Array.isArray(detail.checks)) {
            setChecks(detail.checks);
            setFailure(detail);
            return;
          }
        }
        const message =
          (body && typeof body.detail === "string" && body.detail) ||
          `${res.status} ${res.statusText}`;
        setRequestError(message);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setRequestError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [scope, attempt]);

  // Reveal-cadence driver: once the headline animation finishes AND
  // we have checks in hand, walk the list at REVEAL_INTERVAL_MS.
  useEffect(() => {
    if (!headlineDone) return;
    if (checks.length === 0) return;
    if (phase !== "running") return;
    setPhase("revealing");
  }, [headlineDone, checks.length, phase]);

  useEffect(() => {
    if (phase !== "revealing") return;
    if (revealed >= checks.length) {
      const allOk = checks.every((c) => c.status === "ok");
      setPhase(allOk ? "all_passed" : "failed");
      return;
    }
    const t = setTimeout(() => setRevealed((n) => n + 1), REVEAL_INTERVAL_MS);
    return () => clearTimeout(t);
  }, [phase, revealed, checks]);

  // 1-second countdown after all-pass — operator sees "Begins phase
  // one in 3 / 2 / 1" before the screen advances. The latch prevents a
  // re-render from re-firing onPass after the timer hits zero.
  useEffect(() => {
    if (phase !== "all_passed") return;
    if (advancedRef.current) return;
    if (countdown <= 0) {
      advancedRef.current = true;
      onPass();
      return;
    }
    const t = setTimeout(() => setCountdown((n) => n - 1), 1000);
    return () => clearTimeout(t);
  }, [phase, countdown, onPass]);

  const visible = checks.slice(0, revealed);

  return (
    <div className="preflight">
      <div className="preflight__panel">
        <div className="preflight__headline">
          <TextScramble
            text="Checking environment"
            durationMs={650}
            onDone={() => setHeadlineDone(true)}
          />
          <span className="preflight__ellipsis">…</span>
        </div>

        {requestError && (
          <div className="preflight__error">{requestError}</div>
        )}

        <ul className="preflight__list">
          {visible.map((c) => (
            <li
              key={c.name}
              className={
                "preflight__row preflight__row--enter " +
                (c.status === "ok"
                  ? "preflight__row--ok"
                  : "preflight__row--fail")
              }
            >
              <span className="preflight__icon">
                {c.status === "ok" ? "✓" : "✗"}
              </span>
              <span className="preflight__name">{prettyName(c.name)}</span>
              <span className="preflight__detail">{c.detail}</span>
            </li>
          ))}
          {phase === "running" &&
            Array.from({ length: Math.max(0, 9 - visible.length) }).map(
              (_, i) => (
                <li
                  key={`pending-${i}`}
                  className="preflight__row preflight__row--pending"
                >
                  <span className="preflight__icon">·</span>
                  <span className="preflight__name">&nbsp;</span>
                </li>
              )
            )}
        </ul>

        {phase === "failed" && failure && (
          <div className="preflight__failure">
            <div className="preflight__failure-title">
              {failureHeadline(failure)}
            </div>
            <div className="preflight__failure-detail">{failure.detail}</div>
            {failure.env_var && (
              <pre className="preflight__failure-cmd">
                export {failure.env_var}=…
              </pre>
            )}
            <div className="preflight__actions">
              <button
                type="button"
                className="btn btn--fit"
                onClick={() => setAttempt((n) => n + 1)}
              >
                <span className="btn__cap">Retry</span>
                <span className="btn__key">[↵]</span>
              </button>
              <button
                type="button"
                className="btn"
                onClick={onCancel}
              >
                <span className="btn__cap">Back to query</span>
              </button>
            </div>
          </div>
        )}

        {phase === "all_passed" && (
          <div className="preflight__countdown" aria-live="polite">
            begins phase one in {Math.max(1, countdown)}
          </div>
        )}
      </div>
    </div>
  );
}

function prettyName(name: string): string {
  // The backend's machine names are deliberately compact; the UI
  // surface gets a more human label per check.
  switch (name) {
    case "users.yaml":
      return "users.yaml";
    case "scopes.yaml":
      return "scopes.yaml";
    case "embed_service":
      return "Embed service";
    case "milvus.collection":
      return "Milvus collection";
    case "postgres":
      return "Postgres";
    case "llm.reflection":
      return "Reflection LLM key";
    case "llm.refine.derive":
      return "Refine derive LLM key";
    case "llm.refine.judge":
      return "Refine judge LLM key";
    case "llm.span_extractor":
      return "Span extractor LLM key";
    default:
      return name;
  }
}

function failureHeadline(failure: PreflightFailureDetail): string {
  if (failure.code === "MISSING_LLM_KEY" && failure.env_var) {
    return `Missing ${failure.env_var}`;
  }
  return `Failed: ${prettyName(failure.phase)}`;
}
