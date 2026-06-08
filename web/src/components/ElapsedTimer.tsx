import { useEffect, useRef, useState } from "react";

/**
 * Displays seconds elapsed since mount, formatted as ``X.Ys``. Reuses
 * the ``.turn-overlay__elapsed`` namespace so any progress overlay
 * gets a consistent look. Pattern lifted from :class:`TurnAdvanceOverlay`
 * (which keeps its own inline timer because it also drives the stage-
 * progression overTime branch).
 *
 * Ticks every 100ms by default, 500ms when the operator's system has
 * ``prefers-reduced-motion`` enabled.
 */
export function ElapsedTimer() {
  const startedAt = useRef<number>(Date.now());
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    const prefersReducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const tick = prefersReducedMotion ? 500 : 100;
    const id = window.setInterval(() => {
      setElapsedMs(Date.now() - startedAt.current);
    }, tick);
    return () => window.clearInterval(id);
  }, []);

  return (
    <p className="turn-overlay__elapsed">{`${(elapsedMs / 1000).toFixed(1)}s`}</p>
  );
}
