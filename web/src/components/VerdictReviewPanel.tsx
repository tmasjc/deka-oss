import { useMemo, useState } from "react";
import { formatApiError } from "../api/client";
import {
  useFinalize,
  useRefineDiscard,
  useVerdicts,
} from "../hooks/useRefine";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";
import { ChunkCard } from "./ChunkCard";
import type { RefineSummary, Verdict } from "../types";

type Props = {
  sid: string;
  onFinalized: (summary: RefineSummary) => void;
  // "revisit" hides Finalize / Discard and adds a Back affordance so
  // the operator can re-read verdicts from the DONE summary without
  // mutating the session. Defaults to "review".
  mode?: "review" | "revisit";
  onBack?: () => void;
};

export function VerdictReviewPanel({
  sid,
  onFinalized,
  mode = "review",
  onBack,
}: Props) {
  const verdicts = useVerdicts(sid, true);
  const finalize = useFinalize(sid);
  const discard = useRefineDiscard(sid);
  const [cursor, setCursor] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const items: Verdict[] = useMemo(() => verdicts.data ?? [], [verdicts.data]);
  const safeCursor = items.length ? Math.min(cursor, items.length - 1) : 0;
  const current = items[safeCursor];

  const isReview = mode === "review";
  const canFinalize =
    isReview && items.length > 0 && !finalize.isPending && !discard.isPending;
  const canDiscard =
    isReview && items.length > 0 && !finalize.isPending && !discard.isPending;

  function onFinalize() {
    setError(null);
    finalize.mutate(undefined, {
      onSuccess: (summary) => onFinalized(summary),
      onError: (err) => setError(formatApiError(err) ?? "Finalize failed."),
    });
  }

  function onDiscard() {
    setError(null);
    discard.mutate(undefined, {
      onError: (err) => setError(formatApiError(err) ?? "Discard failed."),
    });
  }

  // Panel-owned keybindings. Rating-level shortcuts are gated to phase
  // === TUNING so j/k/Enter/d here don't collide with the rating loop.
  useKeyboardShortcuts(
    {
      // j = backward (-1), k = forward (+1) to match phase-1 Rating
      // bindings. Arrow keys mirror the same direction as their letter.
      j: () => {
        if (items.length > 0) setCursor((c) => Math.max(c - 1, 0));
      },
      arrowdown: () => {
        if (items.length > 0) setCursor((c) => Math.max(c - 1, 0));
      },
      k: () => {
        if (items.length > 0)
          setCursor((c) => Math.min(c + 1, items.length - 1));
      },
      arrowup: () => {
        if (items.length > 0)
          setCursor((c) => Math.min(c + 1, items.length - 1));
      },
      d: () => {
        if (canDiscard) onDiscard();
      },
      enter: () => {
        if (canFinalize) onFinalize();
      },
      escape: () => {
        if (!isReview && onBack) onBack();
      },
      b: () => {
        if (!isReview && onBack) onBack();
      },
    },
    true,
  );

  if (verdicts.isLoading) {
    return <div className="loading">Loading verdicts…</div>;
  }
  if (verdicts.isError || !current) {
    return (
      <div className="loading">
        {formatApiError(verdicts.error) ?? "No verdicts to review."}
      </div>
    );
  }

  return (
    <section className="verdict-review">
      <div className="verdict-review__bar">
        <span>
          {safeCursor + 1} / {items.length}
        </span>
        <span
          className={
            "verdict-review__verdict-pill verdict-review__verdict-pill--" +
            current.verdict
          }
        >
          {current.verdict}
        </span>
        <span>decile {current.decile}</span>
        {current.failed_check && (
          <span style={{ color: "var(--muted)" }}>
            failed: <code>{current.failed_check}</code>
          </span>
        )}
      </div>

      <div style={{ fontSize: 12, color: "var(--muted)" }}>
        pk <code>{String(current.pk)}</code> · {current.reason}
      </div>

      <ChunkCard
        content={current.chunk_content}
        spanLines={current.evidence_line_indices ?? []}
        rating={null}
      />

      {error && <p className="modal__error">{error}</p>}

      <div className="cfg__actions">
        <button
          type="button"
          className="btn"
          onClick={(e) => {
            setCursor(Math.max(safeCursor - 1, 0));
            e.currentTarget.blur();
          }}
          disabled={safeCursor === 0}
        >
          <span className="btn__cap">Prev</span>
          <span className="btn__key">[j]</span>
        </button>
        <button
          type="button"
          className="btn"
          onClick={(e) => {
            setCursor(Math.min(safeCursor + 1, items.length - 1));
            e.currentTarget.blur();
          }}
          disabled={safeCursor === items.length - 1}
        >
          <span className="btn__cap">Next</span>
          <span className="btn__key">[k]</span>
        </button>
        {isReview ? (
          <>
            <button
              type="button"
              className="btn"
              onClick={(e) => {
                onDiscard();
                e.currentTarget.blur();
              }}
              disabled={!canDiscard}
              style={{ marginLeft: "auto" }}
            >
              <span className="btn__cap">
                {discard.isPending ? "Discarding…" : "Discard"}
              </span>
              <span className="btn__key">[d]</span>
            </button>
            <button
              type="button"
              className="btn btn--fit"
              onClick={(e) => {
                onFinalize();
                e.currentTarget.blur();
              }}
              disabled={!canFinalize}
            >
              <span className="btn__cap">
                {finalize.isPending ? "Finalising…" : "Finalize"}
              </span>
              <span className="btn__key">[↵]</span>
            </button>
          </>
        ) : (
          <button
            type="button"
            className="btn btn--fit"
            onClick={(e) => {
              onBack?.();
              e.currentTarget.blur();
            }}
            disabled={!onBack}
            style={{ marginLeft: "auto" }}
          >
            <span className="btn__cap">Back to summary</span>
            <span className="btn__key">[b]</span>
          </button>
        )}
      </div>
    </section>
  );
}
