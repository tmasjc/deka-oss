import { LoadingHelix } from "./LoadingHelix";

/** Unified "in replay-mode..." overlay used between every replay
 * section transition. Pinned to exactly 3 s by the caller (Rating.tsx
 * races a setTimeout against the network round-trip), so this
 * component itself only renders the visual — it does not own timing.
 *
 * Replaces the per-stage TurnAdvanceOverlay / Harvest / Refine / Apply
 * progress overlays while a session is in replay. The visual matches
 * the live overlays (same LoadingHelix, same surface) so the operator
 * recognises it as "the system is working" — only the text is
 * deliberately unified to signal the replay framing. */
export function ReplayLoader() {
  return (
    <div className="turn-overlay" role="status" aria-live="polite">
      <LoadingHelix />
      <p className="turn-overlay__text">in replay-mode...</p>
    </div>
  );
}
